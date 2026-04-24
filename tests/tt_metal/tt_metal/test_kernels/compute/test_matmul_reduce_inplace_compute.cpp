// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Isolated test compute kernel for matmul_reduce_inplace helper
// (ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp).
//
// The helper reduces `in_out_cb` in place using a column-identity tile in
// `in1_cb`. Used by SDPA to sum rows via a matmul against an all-ones
// column vector tile.
//
// CB layout:
//   c_0  (in_out)      — compute-produced in/out CB for the reduce
//   c_1  (col_ident)   — single column-identity tile (all ones)
//   c_2  (staging)     — reader-produced input; compute copies to c_0
//   c_16 (out_copy)    — copy of c_0 after reduce, so the writer can drain it
//                        via the standard writer_unary path
//
// Design note: matmul_reduce_inplace requires in_out_cb to be compute-owned
// (T2's tiles_received shadow must track the pushes). A reader-produced CB
// would leave T2's shadow at 0 while L1's counter is already nonzero, so
// T2's push-back would overwrite L1 with too-low a value and cause the
// subsequent cb_wait_front to deadlock. The fix is to have the reader write
// to c_2 (staging) and then have the compute kernel copy c_2 → c_0 before
// calling matmul_reduce_inplace, so that T2 owns the c_0 production history.
//
// Compile-time args:
//   [0] num_subblocks
//   [1] subblock_h
//   [2] subblock_w
//   [3] block_kt
//   [4] total_in_tiles (= num_subblocks * subblock_h * subblock_w)

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "api/compute/tile_move_copy.h"
#include "api/compute/cb_api.h"

void kernel_main() {
    constexpr uint32_t num_subblocks = get_compile_time_arg_val(0);
    constexpr uint32_t subblock_h = get_compile_time_arg_val(1);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(2);
    constexpr uint32_t block_kt = get_compile_time_arg_val(3);
    constexpr uint32_t total_in_tiles = get_compile_time_arg_val(4);

    constexpr uint32_t staging_cb = tt::CBIndex::c_2;  // reader-produced staging
    constexpr uint32_t in_out_cb = tt::CBIndex::c_0;   // compute-owned in/out
    constexpr uint32_t col_ident_cb = tt::CBIndex::c_1;
    constexpr uint32_t out_copy_cb = tt::CBIndex::c_16;

    // Phase 1: copy staging → in_out_cb to establish compute ownership of c_0.
    // After this, T2's tiles_received shadow for c_0 correctly reflects the
    // total_in_tiles pushes, so matmul_reduce_inplace's cb_push_back will
    // write the right value to L1.
    mm_init(in_out_cb, col_ident_cb, in_out_cb);
    copy_tile_to_dst_init_short(staging_cb);
    PACK((pack_reconfig_data_format(in_out_cb)));

    cb_wait_front(staging_cb, total_in_tiles);
    cb_reserve_back(in_out_cb, total_in_tiles);
    for (uint32_t i = 0; i < total_in_tiles; i++) {
        tile_regs_acquire();
        copy_tile(staging_cb, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, in_out_cb);
        tile_regs_release();
    }
    cb_pop_front(staging_cb, total_in_tiles);
    cb_push_back(in_out_cb, total_in_tiles);

    // Phase 2: in-place reduce — c_0 is now compute-owned, so T2's counter
    // is in sync with L1. The helper will correctly push-back to c_0 and
    // T0's subsequent cb_wait_front will see the updated L1 counter.
    compute_kernel_lib::matmul_reduce_inplace(in_out_cb, col_ident_cb, num_subblocks, subblock_h, subblock_w, block_kt);

    // Phase 3: copy reduced result from c_0 to c_16 for the writer to drain.
    // Helper ended with srcA configured for in_out_cb (via its internal
    // mm_block_init_short). Re-init for copy_tile, reconfigure pack format.
    copy_tile_to_dst_init_short(in_out_cb);
    PACK((pack_reconfig_data_format(out_copy_cb)));

    // Wait for the reduced-in-place tiles. Counts must match the original
    // input population. If matmul_reduce_inplace left the CB in an unexpected
    // state this wait will hang — that's the in-place regression check.
    cb_wait_front(in_out_cb, total_in_tiles);

    for (uint32_t i = 0; i < total_in_tiles; i++) {
        tile_regs_acquire();
        copy_tile(in_out_cb, i, 0);
        tile_regs_commit();

        cb_reserve_back(out_copy_cb, 1);
        tile_regs_wait();
        pack_tile(0, out_copy_cb);
        tile_regs_release();
        cb_push_back(out_copy_cb, 1);
    }

    cb_pop_front(in_out_cb, total_in_tiles);
}
