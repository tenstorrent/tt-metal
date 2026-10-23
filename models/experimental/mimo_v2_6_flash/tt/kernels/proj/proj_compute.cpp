// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// TRISCs: out[1 x cols] = x[1 x kt] @ w[kt x cols] with the whole K accumulated in Dest, block by block as the
// reader streams w ([rows][cols] per block) and the writer delivers x (block_rows tiles per block).
//
// With k_split > 0 this core holds one K slice of its bank group: it packs its fp32 partials (cols tiles) to cb_part,
// the writers exchange them, and this core sums the group's k_split partials of its out_cols columns (cb_gather, in
// core order, unpacked to Dest in fp32) before packing its output.
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary_sfpu.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t block_rows = get_compile_time_arg_val(4);
    constexpr uint32_t cols = get_compile_time_arg_val(5);
    constexpr uint32_t k_split = get_compile_time_arg_val(6);
    constexpr uint32_t cb_part = get_compile_time_arg_val(7);
    constexpr uint32_t cb_gather = get_compile_time_arg_val(8);
    constexpr uint32_t out_cols = get_compile_time_arg_val(9);
    constexpr uint32_t kt = num_blocks * block_rows;
    constexpr uint32_t block_tiles = block_rows * cols;
    constexpr uint32_t cb_first_out = k_split > 0 ? cb_part : cb_out;

    // matmul unpacks in0 (x) to srcB and in1 (w) to srcA: configure the formats in that order.
    compute_kernel_hw_startup(cb_w, cb_x, cb_first_out);
    matmul_block_init(cb_x, cb_w, 0, cols, 1, kt);

    tile_regs_acquire();
    for (uint32_t b = 0; b < num_blocks; b++) {
        cb_wait_front(cb_x, (b + 1) * block_rows);
        cb_wait_front(cb_w, block_tiles);
        for (uint32_t r = 0; r < block_rows; r++) {
            matmul_block(cb_x, cb_w, b * block_rows + r, r * cols, 0, 0, cols, 1, kt);
        }
        cb_pop_front(cb_w, block_tiles);
    }
    tile_regs_commit();

    if constexpr (k_split > 0) {
        constexpr uint32_t parts = k_split * out_cols;
        cb_reserve_back(cb_part, cols);
        tile_regs_wait();
        for (uint32_t j = 0; j < cols; j++) {
            pack_tile(j, cb_part);
        }
        tile_regs_release();
        cb_push_back(cb_part, cols);
        cb_pop_front(cb_x, kt);

        // Dest j = sum over cores q (in order) of partial q of output column j.
        cb_wait_front(cb_gather, parts);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_gather);
        copy_tile_to_dst_init_short(cb_gather);
        for (uint32_t i = 0; i < parts; i++) {
            copy_tile(cb_gather, i, i);
        }
        add_binary_tile_init();
        for (uint32_t s = 1; s < k_split; s++) {
            for (uint32_t j = 0; j < out_cols; j++) {
                add_binary_tile(j, s * out_cols + j, j);
            }
        }
        tile_regs_commit();
        cb_pop_front(cb_gather, parts);

        pack_reconfig_data_format(cb_out);
        cb_reserve_back(cb_out, out_cols);
        tile_regs_wait();
        for (uint32_t j = 0; j < out_cols; j++) {
            pack_tile(j, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, out_cols);
        return;
    }

    cb_reserve_back(cb_out, cols);
    tile_regs_wait();
    for (uint32_t j = 0; j < cols; j++) {
        pack_tile(j, cb_out);
    }
    tile_regs_release();
    cb_push_back(cb_out, cols);
    cb_pop_front(cb_x, kt);
}
