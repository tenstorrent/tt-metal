// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The append core's compute kernel in program P1 of the attention half (fused/layer_attn/program.py; PLAN.md section
// 17): the new token's cache row of the 8 token rows, from the raw latent row kv_a(x) to what the store holds.
// Standard [32, 32] tiles throughout (rows 0..7 are the token rows, the other rows zero; every operation here is
// row-local), Float32 DEST.
//
//   1. mean of squares of k_pass per row: x^2 of the 16 pass tiles accumulated in one Float32 tile, reduced along
//      the row with the scaler 1 / 512 (the sequence of ttnn's fused_distributed_rmsnorm pre kernel)
//   2. 1 / sqrt(mean + eps) in column 0
//   3. k_pass = x * that (column broadcast) * weight, packed to BF16: cb_lat for the cast, cb_keep for the row
//      writer's copy of the values the cast saw
//   4. k_rope' = k_rope * cos + swap(k_rope) * sin on the two rope tiles (tiles 16, 17 and their swapped twins 18,
//      19 of cb_raw), packed to BF16: cb_rope
//   5. the cast of k_pass to scaled e4m3, per block of 128 values: the sequence of ttnn's per_token_cast_to_fp8
//      compute kernel with a power-of-two scale (block maximum, clamp, / 448, up to a power of two, reciprocal,
//      multiply, pack-untilize to e4m3). The e4m3 packer cuts the mantissa. HY4A_FP8_NEAREST: before the pack half a
//      step of the value's own binade is added to its magnitude (2^(e - 4); 2^-10 below 2^-6, where the format's
//      step is fixed), so the cut rounds half away from zero.
// Outputs for the row writer: cb_e4m3 (4 blocks: 4 pages of [32 rows x 128 bytes] each), cb_scale (4 Float32 tiles,
// the block's scale in column 0), cb_rope (2 BF16 tiles), cb_keep (16 BF16 tiles).
// Tile counts are fixed: nothing here depends on the data or on the positions.
//
// Compile-time args: cb_raw, cb_w, cb_cs, cb_mean, cb_one, cb_sq, cb_stat, cb_inv, cb_tmp, cb_lat, cb_keep, cb_rope,
// cb_tile, cb_abs, cb_scale, cb_invs, cb_e4m3, eps (Float32 bits).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/reduce.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/bcast.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/pack_untilize.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

#ifdef TRISC_MATH
namespace ckernel::sfpu {

// (ttnn's per_token_cast_to_fp8 compute kernel) a positive normal value up to a power of two.
template <int ITERATIONS = 8>
inline void calculate_ceil_power_of_two() {
    for (int d = 0; d < ITERATIONS; ++d) {
        sfpi::vFloat value = sfpi::dst_reg[0];
        sfpi::vInt exponent = sfpi::exexp(value, sfpi::ExponentMode::Biased);
        sfpi::vFloat mantissa = sfpi::setexp(value, 127);
        sfpi::vFloat result = sfpi::setexp(sfpi::vFloat(1.0f), exponent);
        v_if(mantissa != 1.0f) { result = sfpi::setexp(sfpi::vFloat(1.0f), exponent + 1); }
        v_endif;
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// Half an e4m3 step of the value's own binade, added to its magnitude: the packer's cut then rounds half away from
// zero. e4m3 keeps 3 mantissa bits: the step of a value with exponent e is 2^(e - 3); below 2^-6 it is 2^-9.
template <int ITERATIONS = 8>
inline void calculate_e4m3_half_step() {
    for (int d = 0; d < ITERATIONS; ++d) {
        sfpi::vFloat value = sfpi::dst_reg[0];
        sfpi::vInt exponent = sfpi::exexp(value, sfpi::ExponentMode::Biased);
        v_if(exponent < 121) { exponent = 121; }
        v_endif;
        sfpi::vFloat half = sfpi::setexp(sfpi::vFloat(1.0f), exponent - 4);
        v_if(value < 0.0f) { half = -half; }
        v_endif;
        sfpi::dst_reg[0] = value + half;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu

inline void ceil_power_of_two_tile(uint32_t idst) {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_ceil_power_of_two, (8 /* ITERATIONS */), idst, VectorMode::RC);
}

inline void e4m3_half_step_tile(uint32_t idst) {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_e4m3_half_step, (8 /* ITERATIONS */), idst, VectorMode::RC);
}
#endif

void kernel_main() {
    constexpr uint32_t cb_raw = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_cs = get_compile_time_arg_val(2);
    constexpr uint32_t cb_mean = get_compile_time_arg_val(3);
    constexpr uint32_t cb_one = get_compile_time_arg_val(4);
    constexpr uint32_t cb_sq = get_compile_time_arg_val(5);
    constexpr uint32_t cb_stat = get_compile_time_arg_val(6);
    constexpr uint32_t cb_inv = get_compile_time_arg_val(7);
    constexpr uint32_t cb_tmp = get_compile_time_arg_val(8);
    constexpr uint32_t cb_lat = get_compile_time_arg_val(9);
    constexpr uint32_t cb_keep = get_compile_time_arg_val(10);
    constexpr uint32_t cb_rope = get_compile_time_arg_val(11);
    constexpr uint32_t cb_tile = get_compile_time_arg_val(12);
    constexpr uint32_t cb_abs = get_compile_time_arg_val(13);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(14);
    constexpr uint32_t cb_invs = get_compile_time_arg_val(15);
    constexpr uint32_t cb_e4m3 = get_compile_time_arg_val(16);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(17);

    constexpr uint32_t PASS = 16, ROPE = 2, BLOCK = 4, BLOCKS = PASS / BLOCK;
    constexpr uint32_t clamp_min_bits = 0x38D1B717u;  // 1.0e-4f
    constexpr uint32_t clamp_max_bits = 0x7F61B1E6u;  // 3.0e38f
    constexpr uint32_t inv_448_bits = 0x3B124925u;    // 1.0f / 448.0f

    compute_kernel_hw_startup(cb_raw, cb_raw, cb_sq);
    cb_wait_front(cb_mean, 1);
    cb_wait_front(cb_one, 1);
    cb_wait_front(cb_w, PASS);
    cb_wait_front(cb_cs, 4);
    cb_wait_front(cb_raw, PASS + 2 * ROPE);

    // ---- 1. sum over the 16 pass tiles of x^2, element by element, in one Float32 tile (packer L1 accumulation)
    reconfig_data_format(cb_raw, cb_raw);
    pack_reconfig_data_format(cb_sq);
    PACK((llk_pack_reconfig_l1_acc(0)));
    mul_init(cb_raw, cb_raw);
    cb_reserve_back(cb_sq, 1);
    for (uint32_t b = 0; b < PASS; b += BLOCK) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < BLOCK; ++i) {
            mul_tiles(cb_raw, cb_raw, b + i, b + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < BLOCK; ++i) {
            pack_tile<true>(i, cb_sq, 0);
            if (b == 0 && i == 0) {
                PACK((llk_pack_reconfig_l1_acc(1)));
            }
        }
        tile_regs_release();
    }
    cb_push_back(cb_sq, 1);
    PACK((llk_pack_reconfig_l1_acc(0)));
    //      ... and along the row, times 1 / 512: the mean of squares in column 0
    compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_sq, cb_mean, cb_stat>(
        compute_kernel_lib::ReduceInputBlockShape::single());

    // ---- 2. 1 / sqrt(mean + eps)
    cb_wait_front(cb_stat, 1);
    reconfig_data_format_srca(cb_stat);
    pack_reconfig_data_format(cb_inv);
    copy_init(cb_stat);
    cb_reserve_back(cb_inv, 1);
    tile_regs_acquire();
    copy_tile(cb_stat, 0, 0);
    binop_with_scalar_tile_init();
    add_unary_tile(0, eps_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_inv);
    tile_regs_release();
    cb_push_back(cb_inv, 1);
    cb_pop_front(cb_stat, 1);

    // ---- 3. k_pass = x * (1 / rms of the row) * weight, 4 tiles at a time
    cb_wait_front(cb_inv, 1);
    cb_reserve_back(cb_lat, PASS);
    cb_reserve_back(cb_keep, PASS);
    for (uint32_t b = 0; b < PASS; b += BLOCK) {
        reconfig_data_format(cb_raw, cb_inv);
        pack_reconfig_data_format(cb_tmp);
        mul_bcast_cols_init(cb_raw, cb_inv);
        cb_reserve_back(cb_tmp, BLOCK);
        tile_regs_acquire();
        for (uint32_t i = 0; i < BLOCK; ++i) {
            mul_tiles_bcast_cols(cb_raw, cb_inv, b + i, 0, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < BLOCK; ++i) {
            pack_tile(i, cb_tmp);
        }
        tile_regs_release();
        cb_push_back(cb_tmp, BLOCK);

        cb_wait_front(cb_tmp, BLOCK);
        reconfig_data_format(cb_tmp, cb_w);
        pack_reconfig_data_format(cb_lat);
        mul_init(cb_tmp, cb_w);
        tile_regs_acquire();
        for (uint32_t i = 0; i < BLOCK; ++i) {
            mul_tiles(cb_tmp, cb_w, i, b + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < BLOCK; ++i) {
            pack_tile<true>(i, cb_lat, b + i);
            pack_tile<true>(i, cb_keep, b + i);
        }
        tile_regs_release();
        cb_pop_front(cb_tmp, BLOCK);
    }
    cb_push_back(cb_lat, PASS);
    cb_push_back(cb_keep, PASS);
    cb_pop_front(cb_inv, 1);

    // ---- 4. the rotation of k_rope: tiles 16, 17 times cos, their swapped twins 18, 19 times sin
    reconfig_data_format(cb_raw, cb_cs);
    pack_reconfig_data_format(cb_rope);
    mul_init(cb_raw, cb_cs);
    cb_reserve_back(cb_rope, ROPE);
    tile_regs_acquire();
    for (uint32_t t = 0; t < ROPE; ++t) {
        mul_tiles(cb_raw, cb_cs, PASS + t, t, t);                       // x cos
        mul_tiles(cb_raw, cb_cs, PASS + ROPE + t, ROPE + t, ROPE + t);  // swap(x) sin
    }
    add_binary_tile_init();
    for (uint32_t t = 0; t < ROPE; ++t) {
        add_binary_tile(t, ROPE + t, t);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t t = 0; t < ROPE; ++t) {
        pack_tile(t, cb_rope);
    }
    tile_regs_release();
    cb_push_back(cb_rope, ROPE);
    cb_pop_front(cb_raw, PASS + 2 * ROPE);
    cb_pop_front(cb_w, PASS);
    cb_pop_front(cb_cs, 4);
    cb_pop_front(cb_mean, 1);

    // ---- 5. the cast (ttnn per_token_cast_to_fp8's compute kernel, tiled input, one block row of 4 tiles per block)
    compute_kernel_hw_startup(cb_abs, cb_e4m3);
    cb_wait_front(cb_lat, PASS);
    for (uint32_t blk = 0; blk < BLOCKS; ++blk) {
        // the block's tiles into Float32
        reconfig_data_format_srca(cb_lat);
        pack_reconfig_data_format(cb_tile);
        copy_init(cb_lat);
        cb_reserve_back(cb_tile, BLOCK);
        for (uint32_t k = 0; k < BLOCK; ++k) {
            tile_regs_acquire();
            copy_tile(cb_lat, blk * BLOCK + k, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tile);
            tile_regs_release();
        }
        cb_push_back(cb_tile, BLOCK);

        // block maximum -> scale (column 0) and 1 / scale
        cb_wait_front(cb_tile, BLOCK);
        reconfig_full_operand_srca(cb_tile);
        pack_reconfig_data_format(cb_abs);
        copy_init(cb_tile);
        cb_reserve_back(cb_abs, BLOCK);
        abs_tile_init();
        for (uint32_t k = 0; k < BLOCK; ++k) {
            tile_regs_acquire();
            copy_tile(cb_tile, k, 0);
            abs_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_abs);
            tile_regs_release();
        }
        cb_push_back(cb_abs, BLOCK);

        cb_wait_front(cb_abs, BLOCK);
        cb_reserve_back(cb_scale, 1);
        cb_reserve_back(cb_invs, 1);
        tile_regs_acquire();
        reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_abs, cb_one, cb_scale);
        for (uint32_t k = 0; k < BLOCK; ++k) {
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_abs, cb_one, k, 0, k);
        }
        reduce_uninit();
        binary_max_tile_init();
        for (uint32_t i = 1; i < BLOCK; ++i) {
            binary_max_tile(0, i, 0);
        }
        clamp_tile_init();
        clamp_tile(0, clamp_min_bits, clamp_max_bits);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_448_bits);
        MATH((ceil_power_of_two_tile(0)));
        copy_dest_values_init();
        copy_dest_values<DataFormat::Float32>(0, 1);
        recip_tile_init();
        recip_tile(1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_scale);
        pack_tile(1, cb_invs);
        tile_regs_release();
        cb_push_back(cb_scale, 1);
        cb_push_back(cb_invs, 1);
        cb_pop_front(cb_abs, BLOCK);

        // divide and pack-untilize to e4m3
        reconfig_data_format(cb_tile, cb_invs);
        mul_bcast_cols_init(cb_tile, cb_invs);
        pack_untilize_dest_init<BLOCK, BLOCK>(cb_e4m3);
        cb_wait_front(cb_invs, 1);
        cb_reserve_back(cb_e4m3, BLOCK);
        tile_regs_acquire();
        for (uint32_t k = 0; k < BLOCK; ++k) {
            mul_tiles_bcast_cols(cb_tile, cb_invs, k, 0, k);
        }
#ifdef HY4A_FP8_NEAREST
        for (uint32_t k = 0; k < BLOCK; ++k) {
            MATH((e4m3_half_step_tile(k)));
        }
#endif
        tile_regs_commit();
        tile_regs_wait();
        pack_untilize_dest<BLOCK>(cb_e4m3);
        tile_regs_release();
        cb_push_back(cb_e4m3, BLOCK);
        cb_pop_front(cb_tile, BLOCK);
        cb_pop_front(cb_invs, 1);
        pack_untilize_uninit(cb_e4m3);
    }
    cb_pop_front(cb_lat, PASS);
    cb_pop_front(cb_one, 1);
}
