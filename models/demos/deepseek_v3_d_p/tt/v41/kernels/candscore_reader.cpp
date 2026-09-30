// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the V4.1 candidate scoring (candscore_compute.cpp, tt/v41/indexer_kernels.py candidate_scores).
//
// Per query row i of this core: its K candidate block ids (uint32, row-major [R, K]), its 32 heads' query
// (bf16 row-major [R, 32 * 128]: one [32 heads, 128] row-major block, pushed as 4 tile-sized pages for a tilize),
// its 32 head weights (bf16 row-major [R, 32], pushed as row 0 of a 32 x 32 row-major page whose other rows are
// zero) and then, per tile-row of 32 candidate rows (4 blocks of 8 in rank order), the 32 index-K rows as a
// [32, 128] row-major block: one 2 KB read per block from the blocked index-K (bf16 row-major [T / 8, 8 * 128], one
// page per block of 8 rows: whole-block reads, as random 256-byte row reads are DRAM-request bound at ~4x the
// bandwidth time). A sentinel id (>= nblocks) reads block 0 instead (finite values; the writer masks it to -inf).
//
// compile_time_args = [cb_qrm, cb_wrm, cb_krm, cb_ids, K, ROWS_PER_READ, TensorAccessorArgs(q)...,
//                      TensorAccessorArgs(w)..., TensorAccessorArgs(ids)..., TensorAccessorArgs(k_blocks)...]
// runtime args      = [q_addr, w_addr, ids_addr, k_addr, row_start, row_count, nblocks]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t ids_addr = get_arg_val<uint32_t>(2);
    const uint32_t k_addr = get_arg_val<uint32_t>(3);
    const uint32_t row_start = get_arg_val<uint32_t>(4);
    const uint32_t row_count = get_arg_val<uint32_t>(5);
    const uint32_t nblocks = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_qrm = get_compile_time_arg_val(0);
    constexpr uint32_t cb_wrm = get_compile_time_arg_val(1);
    constexpr uint32_t cb_krm = get_compile_time_arg_val(2);
    constexpr uint32_t cb_ids = get_compile_time_arg_val(3);
    constexpr uint32_t K = get_compile_time_arg_val(4);              // candidate blocks per query
    constexpr uint32_t ROWS_PER_READ = get_compile_time_arg_val(5);  // tile-rows of K gathered per read barrier
    constexpr auto q_args = TensorAccessorArgs<6>();
    constexpr auto w_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto ids_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto k_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();

    constexpr uint32_t BLOCK = 8;                      // rows per candidate block
    constexpr uint32_t BLOCK_BYTES = BLOCK * 128 * 2;  // one candidate block's index-K rows (one page)
    constexpr uint32_t TILE_ROWS = K * BLOCK / 32;     // tile-rows of candidates per query
    constexpr uint32_t PAGES = 4;                      // tile-sized pages of one [32, 128] block
    constexpr uint32_t Q_BYTES = 32 * 128 * 2;         // 32 heads x 128
    constexpr uint32_t W_BYTES = 32 * 2;               // 32 head weights
    constexpr uint32_t TILE_BYTES = 32 * 32 * 2;
    static_assert(TILE_ROWS % ROWS_PER_READ == 0);

    const auto q = TensorAccessor(q_args, q_addr);
    const auto w = TensorAccessor(w_args, w_addr);
    const auto ids = TensorAccessor(ids_args, ids_addr);
    const auto k = TensorAccessor(k_args, k_addr);

    DataflowBuffer qrm(cb_qrm);
    DataflowBuffer wrm(cb_wrm);
    DataflowBuffer krm(cb_krm);
    DataflowBuffer idsb(cb_ids);
    const uint32_t ids_l1 = idsb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* id = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ids_l1);

    // both pages of cb_wrm: zero rows 1..31 once (row 0 is overwritten per query)
    {
        volatile tt_l1_ptr uint32_t* wz = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wrm.get_write_ptr());
        for (uint32_t i = 0; i < 2 * TILE_BYTES / 4; ++i) {
            wz[i] = 0;
        }
    }

    for (uint32_t row = row_start; row < row_start + row_count; ++row) {
        noc_async_read(ids.get_noc_addr(row, 0), ids_l1, K * 4);
        qrm.reserve_back(PAGES);
        noc_async_read(q.get_noc_addr(row, 0), qrm.get_write_ptr(), Q_BYTES);
        wrm.reserve_back(1);
        noc_async_read(w.get_noc_addr(row, 0), wrm.get_write_ptr(), W_BYTES);
        noc_async_read_barrier();
        qrm.push_back(PAGES);
        wrm.push_back(1);

        for (uint32_t c0 = 0; c0 < TILE_ROWS; c0 += ROWS_PER_READ) {
            krm.reserve_back(PAGES * ROWS_PER_READ);
            uint32_t l1 = krm.get_write_ptr();
            for (uint32_t b = c0 * 4; b < (c0 + ROWS_PER_READ) * 4; ++b) {
                uint32_t c = id[b];
                c = c < nblocks ? c : 0;
                noc_async_read(k.get_noc_addr(c, 0), l1, BLOCK_BYTES);
                l1 += BLOCK_BYTES;
            }
            noc_async_read_barrier();
            krm.push_back(PAGES * ROWS_PER_READ);
        }
    }
}
