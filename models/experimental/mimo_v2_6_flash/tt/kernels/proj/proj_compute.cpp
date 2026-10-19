// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// TRISCs: out[1 x cols] = x[1 x kt] @ w[kt x cols] with the whole K accumulated in Dest, block by block as the
// reader streams w ([rows][cols] per block) and the writer delivers x (block_rows tiles per block).
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t block_rows = get_compile_time_arg_val(4);
    constexpr uint32_t cols = get_compile_time_arg_val(5);
    constexpr uint32_t kt = num_blocks * block_rows;
    constexpr uint32_t block_tiles = block_rows * cols;

    // matmul unpacks in0 (x) to srcB and in1 (w) to srcA: configure the formats in that order.
    compute_kernel_hw_startup(cb_w, cb_x, cb_out);
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

    cb_reserve_back(cb_out, cols);
    tile_regs_wait();
    for (uint32_t j = 0; j < cols; j++) {
        pack_tile(j, cb_out);
    }
    tile_regs_release();
    cb_push_back(cb_out, cols);
    cb_pop_front(cb_x, kt);
}
