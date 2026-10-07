// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Fused fp4 (e2m1 / per-32 e8m0 scale) quantise-dequantise of a bf16 TILE tensor; one 32-column tile row = one scale
// block.
//   A: |x| -> cb_abs   B: amax = row max -> inv = 1/scale, scale (SFPU)   C: q = grid(x * inv)   D: out = q * scale
// All steps are exact (powers of two / small grid values), same values as pf_tune.fp4_fast.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "fp4_sfpu.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_one = get_compile_time_arg_val(1);
    constexpr uint32_t cb_abs = get_compile_time_arg_val(2);
    constexpr uint32_t cb_inv = get_compile_time_arg_val(3);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(4);
    constexpr uint32_t cb_q = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out = get_compile_time_arg_val(6);
    constexpr uint32_t NB = get_compile_time_arg_val(7);
    const uint32_t nblk = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_x, cb_one, cb_out);
    CircularBuffer x(cb_x), one(cb_one), ab(cb_abs), inv(cb_inv), sc(cb_sc), q(cb_q), out(cb_out);
    one.wait_front(1);

    for (uint32_t b = 0; b < nblk; ++b) {
        x.wait_front(NB);

        // A: |x|
        copy_init(cb_x);
        abs_tile_init();
        tile_regs_acquire();
        for (uint32_t j = 0; j < NB; ++j) {
            copy_tile(cb_x, j, j);
            abs_tile(j);
        }
        tile_regs_commit();
        tile_regs_wait();
        ab.reserve_back(NB);
        for (uint32_t j = 0; j < NB; ++j) {
            pack_tile(j, cb_abs);
        }
        ab.push_back(NB);
        tile_regs_release();

        // B: amax -> 1/scale and scale
        ab.wait_front(NB);
        inv.reserve_back(NB);
        sc.reserve_back(NB);
        reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_abs, cb_one, cb_inv);
        tile_regs_acquire();
        for (uint32_t j = 0; j < NB; ++j) {
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_abs, cb_one, j, 0, j);
        }
        abs_tile_init();
        for (uint32_t j = 0; j < NB; ++j) {
            MATH((_llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::fp4_scale_sfpu<0>, j, VectorMode::RC)));
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < NB; ++j) {
            pack_tile(j, cb_inv);
        }
        tile_regs_release();
        tile_regs_acquire();
        for (uint32_t j = 0; j < NB; ++j) {
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_abs, cb_one, j, 0, j);
        }
        abs_tile_init();
        for (uint32_t j = 0; j < NB; ++j) {
            MATH((_llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::fp4_scale_sfpu<1>, j, VectorMode::RC)));
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < NB; ++j) {
            pack_tile(j, cb_sc);
        }
        tile_regs_release();
        reduce_uninit(cb_abs);
        inv.push_back(NB);
        sc.push_back(NB);
        ab.pop_front(NB);

        // C: q = grid(x * inv)
        inv.wait_front(NB);
        mul_bcast_cols_init(cb_x, cb_inv);
        tile_regs_acquire();
        for (uint32_t j = 0; j < NB; ++j) {
            mul_tiles_bcast_cols(cb_x, cb_inv, j, j, j);
        }
        abs_tile_init();
        for (uint32_t j = 0; j < NB; ++j) {
            MATH((_llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::fp4_grid_sfpu, j, VectorMode::RC)));
        }
        tile_regs_commit();
        tile_regs_wait();
        q.reserve_back(NB);
        for (uint32_t j = 0; j < NB; ++j) {
            pack_tile(j, cb_q);
        }
        q.push_back(NB);
        tile_regs_release();
        inv.pop_front(NB);
        x.pop_front(NB);

        // D: out = q * scale
        q.wait_front(NB);
        sc.wait_front(NB);
        mul_bcast_cols_init(cb_q, cb_sc);
        tile_regs_acquire();
        for (uint32_t j = 0; j < NB; ++j) {
            mul_tiles_bcast_cols(cb_q, cb_sc, j, j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        out.reserve_back(NB);
        for (uint32_t j = 0; j < NB; ++j) {
            pack_tile(j, cb_out);
        }
        out.push_back(NB);
        tile_regs_release();
        q.pop_front(NB);
        sc.pop_front(NB);
    }
}
