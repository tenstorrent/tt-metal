// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The key core's compute kernel in program P1 of a full-indexer layer (fused/layer_attn/program.py; PLAN.md section
// 20): the new token's INDEX KEY of the 8 token rows, from the raw key wk(x) to the BFP8 tiles the key store holds.
// Standard [32, 32] tiles throughout (rows 0..7 are the token rows, the other rows zero; every operation here is
// row-local), Float32 DEST. The key is 4 tiles wide ([rope 64, interleaved | pass 64]); tiles 4, 5 of cb_raw are the
// rope tiles with the members of every pair exchanged, which go through the same norm with a weight and bias of
// their own (sign(j) w[j ^ 1], sign(j) b[j ^ 1], sign = -1 on even columns), so that after the norm they are the
// second operand of the rotation, (-k[2 i + 1], k[2 i]).
//
//   1. mean over the 128 dims per row: the library reduce over the row of 4 tiles with the scaler 1 / 128
//   2. x - mean (column broadcast) for the 6 tiles
//   3. variance: (x - mean)^2 of the 4 key tiles, the library reduce over the row with 1 / 128
//   4. 1 / sqrt(variance + eps) in column 0
//   5. k = (x - mean) * that (column broadcast) * weight + bias, packed to BF16
//   6. the rotation of the rope dims: tiles 0, 1 times cos + their twins 4, 5 times sin; tiles 2, 3 as they are;
//      packed to BF16: cb_rot for the matmul, cb_keep for the row writer's copy of the values the matmul saw
//   7. the Hadamard basis: [rows, 128] @ H [128, 128] (16 BF16 tiles), 4 output tiles, packed to BFP8: the packer's
//      exponent per face row of 16 values is the store's format (a token row shares no exponent with another row)
// Outputs for the row writer: cb_bfp (4 BFP8 tiles), cb_keep (4 BF16 tiles).
// Tile counts are fixed: nothing here depends on the data or on the positions.
//
// Compile-time args: cb_raw, cb_sum, cb_wb, cb_cs, cb_h, cb_mean, cb_stat, cb_cen, cb_sq, cb_var, cb_inv, cb_tmp,
// cb_nw, cb_k, cb_rot, cb_keep, cb_bfp, eps (Float32 bits).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/reduce.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/matmul.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

void kernel_main() {
    constexpr uint32_t cb_raw = get_compile_time_arg_val(0);
    constexpr uint32_t cb_sum = get_compile_time_arg_val(1);
    constexpr uint32_t cb_wb = get_compile_time_arg_val(2);
    constexpr uint32_t cb_cs = get_compile_time_arg_val(3);
    constexpr uint32_t cb_h = get_compile_time_arg_val(4);
    constexpr uint32_t cb_mean = get_compile_time_arg_val(5);
    constexpr uint32_t cb_stat = get_compile_time_arg_val(6);
    constexpr uint32_t cb_cen = get_compile_time_arg_val(7);
    constexpr uint32_t cb_sq = get_compile_time_arg_val(8);
    constexpr uint32_t cb_var = get_compile_time_arg_val(9);
    constexpr uint32_t cb_inv = get_compile_time_arg_val(10);
    constexpr uint32_t cb_tmp = get_compile_time_arg_val(11);
    constexpr uint32_t cb_nw = get_compile_time_arg_val(12);
    constexpr uint32_t cb_k = get_compile_time_arg_val(13);
    constexpr uint32_t cb_rot = get_compile_time_arg_val(14);
    constexpr uint32_t cb_keep = get_compile_time_arg_val(15);
    constexpr uint32_t cb_bfp = get_compile_time_arg_val(16);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(17);

    constexpr uint32_t KEY = 4, ROPE = 2, ALL = KEY + ROPE;
    // tiles in DEST at once: a Float32 DEST holds 4 tiles of [32, 32] (half of the register is the math thread's:
    // ckernel.h get_dest_max_tiles, SyncHalf with 32-bit accumulation); the 6 tiles go through in two batches of 3
    constexpr uint32_t BATCH = 3;

    compute_kernel_hw_startup(cb_raw, cb_raw, cb_stat);
    cb_wait_front(cb_mean, 1);
    cb_wait_front(cb_wb, 2 * ALL);
    cb_wait_front(cb_cs, 4);
    cb_wait_front(cb_h, 4 * KEY);
    cb_wait_front(cb_raw, ALL);

    // ---- 1. the mean of the row over the key's 4 tiles, in column 0 (the reduce pops cb_sum tile by tile)
    compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_sum, cb_mean, cb_stat>(
        compute_kernel_lib::ReduceInputBlockShape::row(KEY));

    // ---- 2. x - mean, the 6 tiles
    cb_wait_front(cb_stat, 1);
    reconfig_data_format(cb_raw, cb_stat);
    pack_reconfig_data_format(cb_cen);
    sub_bcast_cols_init(cb_raw, cb_stat);
    cb_reserve_back(cb_cen, ALL);
    for (uint32_t b = 0; b < ALL; b += BATCH) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < BATCH; ++i) {
            sub_tiles_bcast_cols(cb_raw, cb_stat, b + i, 0, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < BATCH; ++i) {
            pack_tile<true>(i, cb_cen, b + i);
        }
        tile_regs_release();
    }
    cb_push_back(cb_cen, ALL);
    cb_pop_front(cb_stat, 1);
    cb_pop_front(cb_raw, ALL);

    // ---- 3. the variance: the squares of the 4 key tiles, reduced along the row with 1 / 128
    cb_wait_front(cb_cen, ALL);
    reconfig_data_format(cb_cen, cb_cen);
    pack_reconfig_data_format(cb_sq);
    mul_init(cb_cen, cb_cen);
    cb_reserve_back(cb_sq, KEY);
    tile_regs_acquire();
    for (uint32_t i = 0; i < KEY; ++i) {
        mul_tiles(cb_cen, cb_cen, i, i, i);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < KEY; ++i) {
        pack_tile<true>(i, cb_sq, i);
    }
    tile_regs_release();
    cb_push_back(cb_sq, KEY);
    compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_sq, cb_mean, cb_var>(
        compute_kernel_lib::ReduceInputBlockShape::row(KEY));

    // ---- 4. 1 / sqrt(variance + eps)
    cb_wait_front(cb_var, 1);
    reconfig_data_format_srca(cb_var);
    pack_reconfig_data_format(cb_inv);
    copy_init(cb_var);
    cb_reserve_back(cb_inv, 1);
    tile_regs_acquire();
    copy_tile(cb_var, 0, 0);
    binop_with_scalar_tile_init();
    add_unary_tile(0, eps_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_inv);
    tile_regs_release();
    cb_push_back(cb_inv, 1);
    cb_pop_front(cb_var, 1);

    // ---- 5. k = (x - mean) / std * weight + bias, the 6 tiles, in three passes
    cb_wait_front(cb_inv, 1);
    reconfig_data_format(cb_cen, cb_inv);
    pack_reconfig_data_format(cb_tmp);
    mul_bcast_cols_init(cb_cen, cb_inv);
    cb_reserve_back(cb_tmp, ALL);
    for (uint32_t b = 0; b < ALL; b += BATCH) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < BATCH; ++i) {
            mul_tiles_bcast_cols(cb_cen, cb_inv, b + i, 0, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < BATCH; ++i) {
            pack_tile<true>(i, cb_tmp, b + i);
        }
        tile_regs_release();
    }
    cb_push_back(cb_tmp, ALL);
    cb_pop_front(cb_cen, ALL);
    cb_pop_front(cb_inv, 1);

    cb_wait_front(cb_tmp, ALL);
    reconfig_data_format(cb_tmp, cb_wb);
    pack_reconfig_data_format(cb_nw);
    mul_init(cb_tmp, cb_wb);
    cb_reserve_back(cb_nw, ALL);
    for (uint32_t b = 0; b < ALL; b += BATCH) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < BATCH; ++i) {
            mul_tiles(cb_tmp, cb_wb, b + i, b + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < BATCH; ++i) {
            pack_tile<true>(i, cb_nw, b + i);
        }
        tile_regs_release();
    }
    cb_push_back(cb_nw, ALL);
    cb_pop_front(cb_tmp, ALL);

    cb_wait_front(cb_nw, ALL);
    reconfig_data_format(cb_nw, cb_wb);
    pack_reconfig_data_format(cb_k);
    add_init(cb_nw, cb_wb);
    cb_reserve_back(cb_k, ALL);
    for (uint32_t b = 0; b < ALL; b += BATCH) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < BATCH; ++i) {
            add_tiles(cb_nw, cb_wb, b + i, ALL + b + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < BATCH; ++i) {
            pack_tile<true>(i, cb_k, b + i);
        }
        tile_regs_release();
    }
    cb_push_back(cb_k, ALL);
    cb_pop_front(cb_nw, ALL);
    cb_pop_front(cb_wb, 2 * ALL);

    // ---- 6. the rotation of the rope tiles 0, 1 (their twins 4, 5 times sin); the pass tiles 2, 3 as they are
    cb_wait_front(cb_k, ALL);
    reconfig_data_format(cb_k, cb_cs);
    pack_reconfig_data_format(cb_rot);
    mul_init(cb_k, cb_cs);
    cb_reserve_back(cb_rot, KEY);
    cb_reserve_back(cb_keep, KEY);
    tile_regs_acquire();
    for (uint32_t t = 0; t < ROPE; ++t) {
        mul_tiles(cb_k, cb_cs, t, t, t);                      // k cos
        mul_tiles(cb_k, cb_cs, KEY + t, ROPE + t, ROPE + t);  // swap(k) sin
    }
    add_binary_tile_init();
    for (uint32_t t = 0; t < ROPE; ++t) {
        add_binary_tile(t, ROPE + t, t);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t t = 0; t < ROPE; ++t) {
        pack_tile<true>(t, cb_rot, t);
        pack_tile<true>(t, cb_keep, t);
    }
    tile_regs_release();
    reconfig_data_format_srca(cb_k);
    copy_init(cb_k);
    tile_regs_acquire();
    for (uint32_t t = ROPE; t < KEY; ++t) {
        copy_tile(cb_k, t, t);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t t = ROPE; t < KEY; ++t) {
        pack_tile<true>(t, cb_rot, t);
        pack_tile<true>(t, cb_keep, t);
    }
    tile_regs_release();
    cb_push_back(cb_rot, KEY);
    cb_push_back(cb_keep, KEY);
    cb_pop_front(cb_k, ALL);
    cb_pop_front(cb_cs, 4);
    cb_pop_front(cb_mean, 1);

    // ---- 7. the Hadamard basis: output tile c = sum over k of key tile k @ H tile (k, c); packed to BFP8
    cb_wait_front(cb_rot, KEY);
    reconfig_data_format(cb_h, cb_rot);
    pack_reconfig_data_format(cb_bfp);
    matmul_init(cb_rot, cb_h);
    cb_reserve_back(cb_bfp, KEY);
    for (uint32_t c = 0; c < KEY; ++c) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < KEY; ++k) {
            matmul_tiles(cb_rot, cb_h, k, KEY * k + c, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_bfp);
        tile_regs_release();
    }
    cb_push_back(cb_bfp, KEY);
    cb_pop_front(cb_rot, KEY);
    cb_pop_front(cb_h, 4 * KEY);
}
