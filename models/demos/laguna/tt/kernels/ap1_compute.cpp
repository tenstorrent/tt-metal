// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Batch-1 decode attention prologue, compute (Laguna): per-head RMSNorm then HF rotate_half RoPE on a head-major
// [32, 128] block (row = head, 4 tiles). sumsq row-reduced over the 4 tiles with a 1/128 scaler (= mean), + eps,
// rsqrt (column vector) -> x * inv (column broadcast) * w (row broadcast) -> y; RoPE over the first rd_tiles tiles:
// out_j = y_j * cos_j -/+ y_partner * sin_j (partner = j +/- rd_tiles/2, minus for the first half); the rest pass.

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_cos = 2, cb_sin = 3, cb_scaler = 4;
constexpr uint32_t cb_out = 16, cb_sq = 24, cb_inv = 25, cb_n = 26, cb_y = 27;

void kernel_main() {
    constexpr uint32_t rd_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(1);
    constexpr uint32_t half = rd_tiles / 2;
    compute_kernel_hw_startup(cb_x, cb_x, cb_sq);
    cb_wait_front(cb_x, 4);
    cb_wait_front(cb_w, 4);
    cb_wait_front(cb_scaler, 1);

    // sq_j = x_j * x_j
    reconfig_data_format(cb_x, cb_x);
    pack_reconfig_data_format(cb_sq);
    mul_tiles_init(cb_x, cb_x);
    cb_reserve_back(cb_sq, 4);
    for (uint32_t j = 0; j < 4; ++j) {
        tile_regs_acquire();
        mul_tiles(cb_x, cb_x, j, j, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_sq, j);
        tile_regs_release();
    }
    cb_push_back(cb_sq, 4);

    // inv = rsqrt(sum_j rowsum(sq_j) / 128 + eps), column 0
    cb_wait_front(cb_sq, 4);
    reconfig_data_format(cb_sq, cb_scaler);
    pack_reconfig_data_format(cb_inv);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, cb_inv);
    cb_reserve_back(cb_inv, 1);
    tile_regs_acquire();
    for (uint32_t j = 0; j < 4; ++j) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, j, 0, 0);
    }
    reduce_uninit();
    binop_with_scalar_tile_init();
    add_unary_tile(0, eps_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_inv);
    tile_regs_release();
    cb_push_back(cb_inv, 1);
    cb_pop_front(cb_sq, 4);

    // n_j = x_j * inv (column broadcast)
    cb_wait_front(cb_inv, 1);
    reconfig_data_format(cb_x, cb_inv);
    pack_reconfig_data_format(cb_n);
    mul_bcast_cols_init_short(cb_x, cb_inv);
    cb_reserve_back(cb_n, 4);
    for (uint32_t j = 0; j < 4; ++j) {
        tile_regs_acquire();
        mul_tiles_bcast_cols(cb_x, cb_inv, j, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_n, j);
        tile_regs_release();
    }
    cb_push_back(cb_n, 4);
    cb_pop_front(cb_inv, 1);
    cb_pop_front(cb_x, 4);

    // y_j = n_j * w_j (row broadcast)
    cb_wait_front(cb_n, 4);
    reconfig_data_format(cb_n, cb_w);
    pack_reconfig_data_format(cb_y);
    mul_bcast_rows_init_short(cb_n, cb_w);
    cb_reserve_back(cb_y, 4);
    for (uint32_t j = 0; j < 4; ++j) {
        tile_regs_acquire();
        mul_tiles_bcast_rows(cb_n, cb_w, j, j, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_y, j);
        tile_regs_release();
    }
    cb_push_back(cb_y, 4);
    cb_pop_front(cb_n, 4);
    cb_pop_front(cb_w, 4);

    // RoPE
    cb_wait_front(cb_y, 4);
    cb_wait_front(cb_cos, rd_tiles);
    cb_wait_front(cb_sin, rd_tiles);
    pack_reconfig_data_format(cb_out);
    cb_reserve_back(cb_out, 4);
    for (uint32_t j = 0; j < 4; ++j) {
        tile_regs_acquire();
        if (j < rd_tiles) {
            const uint32_t p = j < half ? j + half : j - half;
            reconfig_data_format(cb_y, cb_cos);
            mul_bcast_rows_init_short(cb_y, cb_cos);
            mul_tiles_bcast_rows(cb_y, cb_cos, j, j, 0);
            reconfig_data_format(cb_y, cb_sin);
            mul_bcast_rows_init_short(cb_y, cb_sin);
            mul_tiles_bcast_rows(cb_y, cb_sin, p, j, 1);
            if (j < half) {
                sub_binary_tile_init();
                sub_binary_tile(0, 1, 0);
            } else {
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
        } else {
            reconfig_data_format_srca(cb_y);
            copy_tile_init(cb_y);
            copy_tile(cb_y, j, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out, j);
        tile_regs_release();
    }
    cb_push_back(cb_out, 4);
    cb_pop_front(cb_y, 4);
    cb_pop_front(cb_cos, rd_tiles);
    cb_pop_front(cb_sin, rd_tiles);
    cb_pop_front(cb_scaler, 1);
}
