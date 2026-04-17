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
//   c_0  (in_out)     — input/output CB (produced by reader, overwritten in place)
//   c_1  (col_ident)  — single column-identity tile (all ones)
//   c_16 (out_copy)   — copy of c_0 after reduce, so the writer can drain it
//                       via the standard writer_unary path
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

    constexpr uint32_t in_out_cb = tt::CBIndex::c_0;
    constexpr uint32_t col_ident_cb = tt::CBIndex::c_1;
    constexpr uint32_t out_copy_cb = tt::CBIndex::c_16;

    // The helper calls mm_block_init_short internally, but we still need the
    // global mm_init once at TU start to set up compute infra. Configure pack
    // for in_out_cb since the helper packs back to it (no internal reconfig).
    mm_init(in_out_cb, col_ident_cb, in_out_cb);

    // Reduce in place: c_0 becomes the reduced result while its tile count
    // stays the same (same num_subblocks * subblock_h * subblock_w tiles).
    compute_kernel_lib::matmul_reduce_inplace(in_out_cb, col_ident_cb, num_subblocks, subblock_h, subblock_w, block_kt);

    // Copy the reduced tiles from in_out_cb into out_copy_cb so the writer
    // (consumes c_16) can drain them. This also serves as the "in-place"
    // invariant check: we expect total_in_tiles tiles still fronted.
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
