// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Qwen context-parallel SDPA reader: both owned ring slices of a mesh column in one program.
// Derived from the stock reader_interleaved.cpp (stock_reader.cpp here), keeping only the path the
// context-parallel prefill runs (causal, paged K / V through a page table, streaming compute, no mask /
// sink / MLA / sliding window / windowed mode / KV chains). The per-Q-chunk reads are those of the stock
// reader; what changes is the work list: instead of a contiguous zigzag range of one slice, each core walks
// an explicit list of items (slice p, head nq, q chunk) and each item carries its slice's Q buffer and
// chunk offset.
//
// Runtime args: q0_addr, q1_addr, k_addr, v_addr, page_table_addr, offset_0, offset_1, n_items, item...
//   offset_p: causal chunk offset of slice p in Q chunks (chunk_start_idx / q_chunk_size of the stock call).
//   item = (p << 16) | (nq << 8) | q_chunk.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "dataflow_common.hpp"

void kernel_main() {
    Noc noc;

    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t NKH = get_compile_time_arg_val(2);
    constexpr uint32_t NVH = get_compile_time_arg_val(3);
    constexpr uint32_t Skt = get_compile_time_arg_val(4);
    constexpr uint32_t valid_Sqt = get_compile_time_arg_val(5);
    constexpr uint32_t valid_Skt = get_compile_time_arg_val(6);
    constexpr uint32_t DHt = get_compile_time_arg_val(7);
    constexpr uint32_t vDHt = get_compile_time_arg_val(8);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(9);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(10);
    constexpr uint32_t num_cores = get_compile_time_arg_val(11);
    constexpr uint32_t block_size_t = get_compile_time_arg_val(12);
    constexpr uint32_t page_table_stick_size = get_compile_time_arg_val(13);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(14);

    constexpr auto q_args = TensorAccessorArgs<15>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto page_table_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_arg_offset = page_table_args.next_compile_time_args_offset();
    constexpr uint32_t cb_q_in = get_compile_time_arg_val(cb_arg_offset + 0);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(cb_arg_offset + 1);
    constexpr uint32_t cb_v_in = get_compile_time_arg_val(cb_arg_offset + 2);
    constexpr uint32_t cb_id_page_table = get_compile_time_arg_val(cb_arg_offset + 3);

    uint32_t argidx = 0;
    const uint32_t q0_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t q1_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t k_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t v_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t page_table_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t offset_0 = get_arg_val<uint32_t>(argidx++);
    const uint32_t offset_1 = get_arg_val<uint32_t>(argidx++);
    const uint32_t n_items = get_arg_val<uint32_t>(argidx++);

    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);

    constexpr uint32_t q_heads_per_k = NQH / NKH;
    constexpr uint32_t q_heads_per_v = NQH / NVH;
    constexpr uint32_t q_num_subblocks = Sq_chunk_t / qk_subblock_h;
    constexpr bool use_q_subblock_push = (q_num_subblocks > 1);

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_cores>();

    const auto q0_reader = TensorAccessor(q_args, q0_addr);
    const auto q1_reader = TensorAccessor(q_args, q1_addr);
    const auto k_reader = TensorAccessor(k_args, k_addr);
    const auto v_reader = TensorAccessor(v_args, v_addr);

    const auto q_tile_shape = TensorTileShape(B, NQH, valid_Sqt, DHt);

    CircularBuffer cb_page_table(cb_id_page_table);

    if (n_items == 0) {
        return;
    }
    // B == 1: one page table row, read once (the stock reader reads it on the first item's batch).
    cb_page_table.reserve_back(1);
    volatile tt_l1_ptr uint32_t* page_table_ptr =
        read_page_table_for_batch(noc, cb_id_page_table, 0, page_table_args, page_table_addr, page_table_stick_size);
    cb_page_table.push_back(1);

    for (uint32_t it = 0; it < n_items; ++it) {
        const uint32_t item = get_arg_val<uint32_t>(argidx + it);
        const uint32_t phase = item >> 16;
        const uint32_t nq = (item >> 8) & 0xFF;
        uint32_t q_chunk = item & 0xFF;
        const uint32_t chunked_q_chunk_offset = phase ? offset_1 : offset_0;
        const auto& q_reader = phase ? q1_reader : q0_reader;
        // Flexible chunked: cap at valid_Skt so we never read past the K / V extent.
        const uint32_t valid_Skt_bound = std::min(chunked_q_chunk_offset * Sq_chunk_t + valid_Sqt, valid_Skt);

        const uint32_t q_row_start_tile = std::min(q_chunk * Sq_chunk_t, valid_Sqt);
        const uint32_t q_row_end_tile = std::min(q_row_start_tile + Sq_chunk_t, valid_Sqt);
        const uint32_t q_row_tile_count = q_row_end_tile - q_row_start_tile;
        uint32_t q_read_tile_id = q_tile_shape.id_of(0, nq, q_row_start_tile, 0);

        if constexpr (!use_q_subblock_push) {
            read_chunk_with_padding<q_tile_bytes>(
                q_reader, cb_q_in, q_read_tile_id, q_row_tile_count, DHt, Sq_chunk_t, DHt, barrier_threshold);
        }

        q_chunk = chunked_q_chunk_offset + q_chunk;
        const uint32_t q_low_idx = q_chunk * Sq_chunk_t;
        const uint32_t q_high_unclamped = q_low_idx + Sq_chunk_t;
        const uint32_t q_high_idx = q_high_unclamped < Skt ? q_high_unclamped : Skt;

        const uint32_t k_head = nq / q_heads_per_k;
        const uint32_t v_head = nq / q_heads_per_v;

        for (uint32_t k_chunk = 0; (k_chunk * Sk_chunk_t) < q_high_idx; ++k_chunk) {
            const uint32_t kv_row_start_tile = std::min(k_chunk * Sk_chunk_t, valid_Skt_bound);
            const uint32_t kv_row_end_tile = std::min(kv_row_start_tile + Sk_chunk_t, valid_Skt_bound);
            const uint32_t kv_row_tile_count = kv_row_end_tile - kv_row_start_tile;

            const uint32_t k_chunk_start_row_num = k_chunk * Sk_chunk_t;
#ifdef CP_SKIP_KV_DRAM
            // Measurement only (cp_step.hpp knobs): hand the compute stale K / V pages without the DRAM reads.
            {
                CircularBuffer cbk(cb_k_in);
                cbk.reserve_back(Sk_chunk_t * DHt);
                cbk.push_back(Sk_chunk_t * DHt);
            }
            (void)kv_row_tile_count;
            (void)k_chunk_start_row_num;
#else
            read_paged_chunk_with_padding<NKH, block_size_t, DHt>(
                k_reader,
                cb_k_in,
                k_head,
                k_chunk_start_row_num,
                kv_row_tile_count,
                DHt,
                Sk_chunk_t,
                DHt,
                k_tile_bytes,
                barrier_threshold,
                page_table_ptr,
                true  // transpose=true for K reads
            );
#endif

            if constexpr (use_q_subblock_push) {
                if (k_chunk == 0) {
                    for (uint32_t q_sub = 0; q_sub < q_num_subblocks; ++q_sub) {
                        read_q_subblock<q_tile_bytes>(
                            q_reader,
                            cb_q_in,
                            q_read_tile_id,
                            q_sub * qk_subblock_h,
                            qk_subblock_h,
                            q_row_tile_count,
                            DHt,
                            DHt,
                            barrier_threshold);
                    }
                }
            }

            const uint32_t kv_chunk_start_row_num = k_chunk * Sk_chunk_t;
#ifdef CP_SKIP_KV_DRAM
            {
                CircularBuffer cbv(cb_v_in);
                cbv.reserve_back(Sk_chunk_t * vDHt);
                cbv.push_back(Sk_chunk_t * vDHt);
            }
            (void)kv_chunk_start_row_num;
#else
            read_paged_chunk_with_padding<NVH, block_size_t, DHt>(
                v_reader,
                cb_v_in,
                v_head,
                kv_chunk_start_row_num,
                kv_row_tile_count,
                vDHt,
                Sk_chunk_t,
                vDHt,
                v_tile_bytes,
                barrier_threshold,
                page_table_ptr,
                false,
                0);
#endif
        }
    }
    cb_page_table.pop_front(1);
}
