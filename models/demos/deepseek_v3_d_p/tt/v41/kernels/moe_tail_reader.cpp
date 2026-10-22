// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the DeepSeek-V4.1 decode MoE tail (tt/v41/moe_tail.py): the weighted sum over the top-k slots of the
// combine buffer for one tile of tokens, replacing ``post_combine_reduce``'s per-token scalar-broadcast chain.
// ``combine`` is row-major bf16 [T, K, H] (one page per (token, slot) row; page t * K + k); this core owns TPC
// consecutive 32-column tiles from tile j0. For each it builds K activation tiles (tile (jj, k) at CB page jj * K + k;
// row t = the slot's row of token t if the slot's expert is in this chip's dispatch group, zero otherwise: a stale
// row of the combine buffer must not reach the MAC) and one score tile per slot (column 0 of row t = the slot's
// weight if on, else 0), in tile layout, so the compute kernel MACs them with a column broadcast: the tiles are
// zeroed by the NoC and only the rows that are on are read from DRAM and copied in.
//
// compile_time_args = [cb_act, cb_score, cb_scratch, T, K, TPC, table words,
//                      TensorAccessorArgs(combine)..., (scores)..., (indices)..., (table)...]
// runtime args      = [combine_addr, scores_addr, indices_addr, table_addr, j0]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    const uint32_t combine_addr = get_arg_val<uint32_t>(0);
    const uint32_t scores_addr = get_arg_val<uint32_t>(1);
    const uint32_t indices_addr = get_arg_val<uint32_t>(2);
    const uint32_t table_addr = get_arg_val<uint32_t>(3);
    const uint32_t j0 = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_act = get_compile_time_arg_val(0);
    constexpr uint32_t cb_score = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(2);
    constexpr uint32_t T = get_compile_time_arg_val(3);
    constexpr uint32_t K = get_compile_time_arg_val(4);
    constexpr uint32_t TPC = get_compile_time_arg_val(5);
    constexpr uint32_t TABLE_WORDS = get_compile_time_arg_val(6);
    constexpr auto combine_args = TensorAccessorArgs<7>();
    constexpr auto scores_args = TensorAccessorArgs<combine_args.next_compile_time_args_offset()>();
    constexpr auto indices_args = TensorAccessorArgs<scores_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr uint32_t STRIDE = 64;     // the scores and indices rows in L1, one 64-byte line each
    constexpr uint32_t SEG = TPC * 64;  // this core's columns of one combine row
    constexpr uint32_t TILE_WORDS = 512;
    constexpr uint32_t OFF_SC = 0;
    constexpr uint32_t OFF_IDX = OFF_SC + 32 * STRIDE;
    constexpr uint32_t OFF_TABLE = OFF_IDX + 32 * STRIDE;
    constexpr uint32_t OFF_ROWS = OFF_TABLE + ((TABLE_WORDS * 4 + 63) & ~63u);

    const auto combine = TensorAccessor(combine_args, combine_addr);
    const auto scores = TensorAccessor(scores_args, scores_addr);
    const auto indices = TensorAccessor(indices_args, indices_addr);
    const auto table = TensorAccessor(table_args, table_addr);
    DataflowBuffer act(cb_act), score(cb_score), scratch(cb_scratch);
    const uint32_t base = (scratch.get_write_ptr() + 63) & ~63u;

    for (uint32_t t = 0; t < T; ++t) {
        noc_async_read(scores.get_noc_addr(t, 0), base + OFF_SC + t * STRIDE, K * 2);
        noc_async_read(indices.get_noc_addr(t, 0), base + OFF_IDX + t * STRIDE, K * 2);
    }
    noc_async_read(table.get_noc_addr(0, 0), base + OFF_TABLE, TABLE_WORDS * 4);
    noc_async_read_barrier();

    volatile tt_l1_ptr uint16_t* sc = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + OFF_SC);
    volatile tt_l1_ptr uint16_t* ix = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + OFF_IDX);
    volatile tt_l1_ptr int32_t* in_group = reinterpret_cast<volatile tt_l1_ptr int32_t*>(base + OFF_TABLE);
    // the rows that are on: the slot's expert is in this chip's dispatch group (the table's -1 marks the others,
    // and the pad rows' sentinel index is the table's trailing -1 column)
    uint32_t on[K];
    for (uint32_t k = 0; k < K; ++k) {
        on[k] = 0;
    }
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t k = 0; k < K; ++k) {
            if (in_group[ix[t * (STRIDE / 2) + k]] >= 0) {
                on[k] |= 1u << t;
                noc_async_read(combine.get_noc_addr(t * K + k, j0 * 64), base + OFF_ROWS + (t * K + k) * SEG, SEG);
            }
        }
    }

    act.reserve_back(K * TPC);
    score.reserve_back(K);
    // the tiles zeroed by the NoC from the core's zero page (every row of every tile; plain reads, as the rows
    // above: ``Noc::async_write_zeros`` is a stateful read on Blackhole and hung the captured step's 13th replay),
    // then only the rows that are on written over them
    const uint64_t zeros = get_noc_addr(my_x[noc_index], my_y[noc_index], MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < K * TPC * TILE_WORDS * 4; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, act.get_write_ptr() + off, MEM_ZEROS_SIZE);
    }
    for (uint32_t off = 0; off < K * TILE_WORDS * 4; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, score.get_write_ptr() + off, MEM_ZEROS_SIZE);
    }
    noc_async_read_barrier();

    volatile tt_l1_ptr uint16_t* sp16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(score.get_write_ptr());
    volatile tt_l1_ptr uint32_t* rows = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + OFF_ROWS);
    volatile tt_l1_ptr uint32_t* ap = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act.get_write_ptr());
    for (uint32_t k = 0; k < K; ++k) {
        for (uint32_t t = 0; t < T; ++t) {
            if (!(on[k] & (1u << t))) {
                continue;
            }
            // score tile k: column 0 of row t
            sp16[k * 1024 + ((t >> 4) << 9) + ((t & 15) << 4)] = sc[t * (STRIDE / 2) + k];
            // activation tile (jj, k): row t is in face (t / 16) * 2 and the one after it, 8 words each
            volatile tt_l1_ptr uint32_t* s = rows + ((t * K + k) * SEG) / 4;
            for (uint32_t jj = 0; jj < TPC; ++jj) {
                volatile tt_l1_ptr uint32_t* d = ap + (jj * K + k) * TILE_WORDS + ((t >> 4) << 8) + ((t & 15) << 3);
                for (uint32_t w = 0; w < 8; ++w) {
                    d[w] = s[jj * 16 + w];
                    d[128 + w] = s[jj * 16 + 8 + w];
                }
            }
        }
    }
    act.push_back(K * TPC);
    score.push_back(K);
}
