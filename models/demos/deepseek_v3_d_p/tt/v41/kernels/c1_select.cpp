// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Local expert selection of the DeepSeek-V4.1 c1 decode MoE (tt/v41/moe_c1.py c1_select): one token row, its
// top-K expert ids and weights replicated on every chip, and this chip's EPC experts (``table``, global ids) ->
// the routed expert FFN's per-expert token counts and the weight matrix of the local weighted sum:
//
//   counts[e] = 1 if e is one of the row's K experts and one of this chip's, else 0      (uint32 [1, E])
//   W[0, 32 * l] = the row's weight of local expert l if the row selected it, else 0    (bf16 TILE [32, EPC * 32])
//
// ``idx`` is router_select's uint16 TILE [T, K] (row 0 the live row; one tile), ``weights`` its bf16 row-major
// [T, K] (page 0 the live row), ``table`` the chip's global_expert_idx_table (uint32, EPC words). One core.
//
// compile_time_args = [cb_scratch, E, K, EPC, TensorAccessorArgs(idx)..., (weights)..., (table)..., (counts)...,
//                      (w)...]
// runtime args      = [idx_addr, weights_addr, table_addr, counts_addr, w_addr]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    const uint32_t idx_addr = get_arg_val<uint32_t>(0);
    const uint32_t weights_addr = get_arg_val<uint32_t>(1);
    const uint32_t table_addr = get_arg_val<uint32_t>(2);
    const uint32_t counts_addr = get_arg_val<uint32_t>(3);
    const uint32_t w_addr = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_scratch = get_compile_time_arg_val(0);
    constexpr uint32_t E = get_compile_time_arg_val(1);
    constexpr uint32_t K = get_compile_time_arg_val(2);
    constexpr uint32_t EPC = get_compile_time_arg_val(3);
    constexpr auto idx_args = TensorAccessorArgs<4>();
    constexpr auto weights_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();
    constexpr auto counts_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE_BYTES = 2048;
    constexpr uint32_t COUNT_BYTES = E * 4;
    constexpr uint32_t TABLE_L1 = (EPC * 4 + 63) & ~63u;

    const auto idx = TensorAccessor(idx_args, idx_addr);
    const auto weights = TensorAccessor(weights_args, weights_addr);
    const auto table = TensorAccessor(table_args, table_addr);
    const auto counts = TensorAccessor(counts_args, counts_addr);
    const auto w = TensorAccessor(w_args, w_addr);
    DataflowBuffer scratch(cb_scratch);
    const uint32_t base = (scratch.get_write_ptr() + 63) & ~63u;
    const uint32_t idx_l1 = base;
    const uint32_t wt_l1 = idx_l1 + TILE_BYTES;
    const uint32_t table_l1 = wt_l1 + 64;
    const uint32_t counts_l1 = table_l1 + TABLE_L1;
    const uint32_t w_l1 = counts_l1 + ((COUNT_BYTES + 63) & ~63u);

    noc_async_read(idx.get_noc_addr(0, 0), idx_l1, TILE_BYTES);
    noc_async_read(weights.get_noc_addr(0, 0), wt_l1, K * 2);
    noc_async_read(table.get_noc_addr(0, 0), table_l1, EPC * 4);
    noc_async_read_barrier();

    volatile tt_l1_ptr uint16_t* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(idx_l1);  // row 0: face 0
    volatile tt_l1_ptr uint16_t* wt = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(wt_l1);
    volatile tt_l1_ptr uint32_t* mine = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table_l1);
    volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_l1);
    volatile tt_l1_ptr uint32_t* wtile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w_l1);

    for (uint32_t e = 0; e < E; ++e) {
        cnt[e] = 0;
    }
    for (uint32_t i = 0; i < EPC * TILE_BYTES / 4; ++i) {
        wtile[i] = 0;
    }
    for (uint32_t j = 0; j < K; ++j) {
        const uint32_t e = ids[j];
        if (e >= E) {
            continue;  // the sentinel
        }
        for (uint32_t l = 0; l < EPC; ++l) {
            if (mine[l] == e) {
                cnt[e] = 1;
                // element (0, 32 l): tile l, face 0, row 0, column 0 = the tile's first 16-bit word
                wtile[l * (TILE_BYTES / 4)] = wt[j];
            }
        }
    }
    noc_async_write(counts_l1, counts.get_noc_addr(0, 0), COUNT_BYTES);
    for (uint32_t l = 0; l < EPC; ++l) {
        noc_async_write(w_l1 + l * TILE_BYTES, w.get_noc_addr(l, 0), TILE_BYTES);
    }
    noc_async_write_barrier();
}
