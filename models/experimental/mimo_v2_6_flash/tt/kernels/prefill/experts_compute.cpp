// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// TRISCs of the grouped prefill experts. Per job the expert's weight slices sit in cb_wgu / cb_wd for all of its
// groups. Per 32-row group: this core's quarter of the group's x rows (cb_xrm, row-major) tilized into cb_xq for the
// writer to share with the team; once the team's 128 x tiles are in cb_x, gate and up of this core's two intermediate
// columns (banks b0, b1) accumulated over the 128 x tiles in fp32 dest (dest 0..3 = gate b0, up b0, gate b1, up b1), h
// = silu(gate) * up * routing weight (dest 0 and 2) -> cb_h_own; once the writer has gathered the team's 8 h tiles into
// cb_h_all, this core's 32 output columns of down (4 per dest pass, K = the 8 h tiles) -> cb_y.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tilize.h"

void kernel_main() {
    constexpr uint32_t cb_plan = get_compile_time_arg_val(0);
    constexpr uint32_t cb_wgu = get_compile_time_arg_val(1);
    constexpr uint32_t cb_wd = get_compile_time_arg_val(2);
    constexpr uint32_t cb_x = get_compile_time_arg_val(3);
    constexpr uint32_t cb_rw = get_compile_time_arg_val(4);
    constexpr uint32_t cb_h_own = get_compile_time_arg_val(5);
    constexpr uint32_t cb_h_all = get_compile_time_arg_val(6);
    constexpr uint32_t cb_y = get_compile_time_arg_val(7);
    constexpr uint32_t kt = get_compile_time_arg_val(8);  // 128
    constexpr uint32_t cb_xrm = get_compile_time_arg_val(9);
    constexpr uint32_t gu_tiles = get_compile_time_arg_val(10);  // per bank: [kt][gate, up]
    constexpr uint32_t wd_tiles = get_compile_time_arg_val(11);  // per bank: [ikt][cols]
    constexpr uint32_t ikt = get_compile_time_arg_val(12);       // intermediate tiles = 8
    constexpr uint32_t cols = get_compile_time_arg_val(13);      // down output columns per bank = 16
    constexpr uint32_t cb_xq = get_compile_time_arg_val(14);
    constexpr uint32_t quarter_tiles = get_compile_time_arg_val(15);  // kt / team size
    constexpr uint32_t out_block = 4;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_wgu, cb_h_own);
    cb_wait_front(cb_plan, 1);
    const uint32_t jobs = read_tile_value(cb_plan, 0, 0);

    for (uint32_t j = 0; j < jobs; j++) {
        const uint32_t groups = read_tile_value(cb_plan, 0, 3 + 3 * j);
        cb_wait_front(cb_wgu, 2 * gu_tiles);
        cb_wait_front(cb_wd, 2 * wd_tiles);
        for (uint32_t i = 0; i < groups; i++) {
            // this core's quarter of x: row-major -> tiles (srcA was last the BFP4 down weights)
            reconfig_data_format_srca(cb_wd, cb_xrm);
            tilize_init(cb_xrm, quarter_tiles, cb_xq);
            cb_wait_front(cb_xrm, quarter_tiles);
            cb_reserve_back(cb_xq, quarter_tiles);
            tilize_block(cb_xrm, quarter_tiles, cb_xq);
            cb_push_back(cb_xq, quarter_tiles);
            cb_pop_front(cb_xrm, quarter_tiles);
            tilize_uninit(cb_xrm, cb_xq);

            // gate | up of both banks, accumulated over the team's K tiles
            reconfig_data_format_srca(cb_xrm, cb_wgu);
            matmul_init(cb_x, cb_wgu);
            cb_wait_front(cb_x, kt);
            tile_regs_acquire();
            for (uint32_t k = 0; k < kt; k++) {
                matmul_tiles(cb_x, cb_wgu, k, 2 * k, 0);
                matmul_tiles(cb_x, cb_wgu, k, 2 * k + 1, 1);
                matmul_tiles(cb_x, cb_wgu, k, gu_tiles + 2 * k, 2);
                matmul_tiles(cb_x, cb_wgu, k, gu_tiles + 2 * k + 1, 3);
            }
            cb_pop_front(cb_x, kt);
            // h = silu(gate) * up * w (w: the rows' routing weights, every column; copied into dest 1 once up is used)
            silu_tile_init();
            silu_tile(0);
            silu_tile(2);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            mul_binary_tile(2, 3, 2);
            cb_wait_front(cb_rw, 1);
            reconfig_data_format_srca(cb_wgu, cb_rw);  // matmul left srcA in the weights' format
            copy_init(cb_rw);
            copy_tile(cb_rw, 0, 1);
            cb_pop_front(cb_rw, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            mul_binary_tile(2, 1, 2);
            tile_regs_commit();
            cb_reserve_back(cb_h_own, 2);
            tile_regs_wait();
            pack_tile(0, cb_h_own);
            pack_tile(2, cb_h_own);
            tile_regs_release();
            cb_push_back(cb_h_own, 2);

            // down: this core's output columns (bank b0: cols b0 + 8m, bank b1: b1 + 8m), K = the team's 8 h tiles
            cb_wait_front(cb_h_all, ikt);
            reconfig_data_format_srca(cb_rw, cb_wd);
            matmul_init(cb_h_all, cb_wd);
            for (uint32_t p = 0; p < 2 * cols / out_block; p++) {
                const uint32_t w0 = (p / (cols / out_block)) * wd_tiles + (p % (cols / out_block)) * out_block;
                tile_regs_acquire();
                for (uint32_t r = 0; r < ikt; r++) {
                    for (uint32_t c = 0; c < out_block; c++) {
                        matmul_tiles(cb_h_all, cb_wd, r, w0 + r * cols + c, c);
                    }
                }
                tile_regs_commit();
                cb_reserve_back(cb_y, out_block);
                tile_regs_wait();
                for (uint32_t c = 0; c < out_block; c++) {
                    pack_tile(c, cb_y);
                }
                tile_regs_release();
                cb_push_back(cb_y, out_block);
            }
            cb_pop_front(cb_h_all, ikt);
        }
        cb_pop_front(cb_wgu, 2 * gu_tiles);
        cb_pop_front(cb_wd, 2 * wd_tiles);
    }
}
