// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reduce_custom.h"
#include "api/compute/binary_max_min.h"
#include "tensor_shape.h"

// Repro kernel: exercises the RUNTIME reduce_block_max_row path (reduce_block_max_row_init_runtime /
// reduce_block_max_row_runtime / reduce_block_max_row_uninit_runtime) — the exact path the streaming
// SDPA uses (via reduce_c_row_group), which the compile-time reduce_block_max_row kernel does NOT cover.
// For a 16x32 tiny tile (num_faces=2) this is where the second k-face (F1) is dropped from the row-max.
void kernel_main() {
    constexpr std::uint32_t qk_im_cb = get_compile_time_arg_val(0);
    constexpr std::uint32_t prev_max_cb = get_compile_time_arg_val(1);
    constexpr std::uint32_t out_max_cb = get_compile_time_arg_val(2);
    constexpr std::uint32_t scale_cb = get_compile_time_arg_val(3);
    constexpr std::uint32_t Sq_chunk_t = get_compile_time_arg_val(4);
    constexpr std::uint32_t Sk_chunk_t = get_compile_time_arg_val(5);
    constexpr std::uint32_t do_eltwise = get_compile_time_arg_val(6);
    // Operand tile geometry: 4 faces for a 32x32 tile, 2 faces for a 16x32 tiny tile.
    constexpr std::uint32_t num_faces = get_compile_time_arg_val(7);

    // Init compute
    compute_kernel_hw_startup<SrcOrder::Reverse>(qk_im_cb, qk_im_cb, qk_im_cb);
    matmul_init(qk_im_cb, qk_im_cb);

    cb_push_back(qk_im_cb, Sq_chunk_t * Sk_chunk_t);
    cb_push_back(scale_cb, 1);
    if (do_eltwise) {
        cb_push_back(prev_max_cb, Sq_chunk_t);
    }

    reconfig_data_format(qk_im_cb, scale_cb);
    pack_reconfig_data_format(out_max_cb);

    constexpr std::uint32_t rows = Sq_chunk_t;
    constexpr std::uint32_t cols = Sk_chunk_t;
    constexpr std::uint32_t num_tiles = rows * cols;
    cb_wait_front(scale_cb, 1);
    cb_wait_front(qk_im_cb, num_tiles);
    cb_reserve_back(out_max_cb, rows);

    constexpr std::uint32_t reduce_dst_idx = 0;
    constexpr std::uint32_t prev_max_dst_idx = 1;

    for (std::uint32_t i = 0; i < rows; i++) {
        tile_regs_acquire();
        // Runtime reduce path (matches SDPA reduce_c_row_group's inner call).
        reduce_block_max_row_init_runtime(out_max_cb, cols, /*respect_trigger=*/false, num_faces);
        reduce_block_max_row_runtime(
            qk_im_cb, scale_cb, i * cols, reduce_dst_idx, /*respect_trigger=*/false, /*overlap_first_half=*/false, num_faces);
        reduce_block_max_row_uninit_runtime(qk_im_cb);

        if (do_eltwise) {
            copy_tile_to_dst_init_short(prev_max_cb);
            copy_tile(prev_max_cb, i, prev_max_dst_idx);
            binary_max_tile_init();
            binary_max_tile(reduce_dst_idx, prev_max_dst_idx, reduce_dst_idx);
        }

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(reduce_dst_idx, out_max_cb);
        tile_regs_release();
    }

    cb_push_back(out_max_cb, rows);

    // Ensure outputs are produced before exiting
    cb_wait_front(out_max_cb, Sq_chunk_t);
}
