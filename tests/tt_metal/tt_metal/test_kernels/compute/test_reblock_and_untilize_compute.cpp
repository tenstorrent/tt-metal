// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Isolated test compute kernel for reblock_and_untilize helper
// (ttnn/cpp/ttnn/kernel_lib/reblock_untilize_helpers.hpp).
//
// Drives the helper directly with a synthetic subblock-major input — no
// matmul involved. Exercises a single row-group per invocation, repeated
// `num_row_groups` times (matches the production caller that loops
// in0_num_subblocks times).
//
// CB layout:
//   c_24 (interm)  — subblock-major input (reader pushes the whole test buffer)
//   c_16 (out)     — untilized row-major output
//
// Compile-time args:
//   [0] num_row_groups      (= in0_num_subblocks)
//   [1] num_subblocks_w     (= in1_num_subblocks)
//   [2] out_subblock_h
//   [3] out_subblock_w
//   [4] out_block_w         (= out_subblock_w * num_subblocks_w)

#include <cstdint>

#include "api/compute/tile_move_copy.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/reblock_untilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t num_row_groups = get_compile_time_arg_val(0);
    constexpr uint32_t num_subblocks_w = get_compile_time_arg_val(1);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(2);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(3);
    constexpr uint32_t out_block_w = get_compile_time_arg_val(4);
    constexpr uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    constexpr uint32_t interm_cb = tt::CBIndex::c_24;
    constexpr uint32_t out_cb = tt::CBIndex::c_16;

    // Bootstrap compute infrastructure for the untilize-only pipeline.
    compute_kernel_hw_startup(interm_cb, out_cb);

    // Helper prereqs: pack_untilize_dest_init + copy_tile_to_dst_init_short
    // must be called once before the first invocation, and pack_untilize_uninit
    // after the last. Mirrors the production caller.
    pack_untilize_dest_init<out_subblock_w, out_block_w>(out_cb);
    copy_tile_to_dst_init_short(interm_cb);

    for (uint32_t g = 0; g < num_row_groups; g++) {
        compute_kernel_lib::reblock_and_untilize<out_subblock_w, out_block_w>(
            num_subblocks_w, out_subblock_num_tiles, out_subblock_h, interm_cb, out_cb);
    }

    pack_untilize_uninit(interm_cb);
}
