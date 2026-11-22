// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused attention glue, head role, compute (fp32 SFPU math). Per item, over its nlag dense pages:
//   y = sum_l x_l * (t_l [+ cs * d_l])          q: nlag = 1, t = q pre-scale; k / v: short conv
//   s = sum(y^2)                                  (reduce, scalar at [0, 0])
//   norm 0 (L2 clip): f = rsqrt(max(s * inv, 1)) [* tau];   norm 1 (RMS): f = rsqrt(s * inv + eps)
//   out = y * f -> cb_out (bf16, or the cache's bfp8: packed per 16-element face row like the cache tiles)
// mass: also pack the far-cache mass page (cb_massin) into cb_mass in the cache format.
#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"

namespace {
constexpr uint32_t cb_x = 0, cb_t = 1, cb_d = 2, cb_scaler = 3, cb_tau = 4, cb_massin = 5;
constexpr uint32_t cb_y = 16, cb_sq = 17, cb_f = 18, cb_out = 19, cb_mass = 20;
// SFPU work on faces 0, 1 only: a head is elements 0..127 of its dense page (face 0); the reader zero-fills the rest
// of x, so y stays 0 there either way (y^2 is still squared over the whole tile, it feeds the full-tile reduce), and
// the norm factor is read at [0, 0] only (scalar broadcast).
constexpr VectorMode R = VectorMode::R;

FORCE_INLINE void load(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}
FORCE_INLINE void put(uint32_t dst, uint32_t cb) {
    cb_reserve_back(cb, 1);
    pack_reconfig_data_format(cb);
    pack_tile(dst, cb);
    cb_push_back(cb, 1);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t nitems = get_named_compile_time_arg_val("nitems");
    constexpr uint32_t nlag = get_named_compile_time_arg_val("nlag");
    constexpr uint32_t dyn = get_named_compile_time_arg_val("dyn");
    constexpr uint32_t has_tau = get_named_compile_time_arg_val("has_tau");
    constexpr uint32_t mass = get_named_compile_time_arg_val("mass");
    constexpr uint32_t norm = get_named_compile_time_arg_val("norm");
    constexpr uint32_t inv = get_named_compile_time_arg_val("inv");  // fp32 bits
    constexpr uint32_t eps = get_named_compile_time_arg_val("eps");  // fp32 bits
    constexpr uint32_t cs = get_named_compile_time_arg_val("cs");    // fp32 bits (dyn only)
    constexpr uint32_t one = 0x3f800000u;
    constexpr uint32_t n = nitems * nlag;

    compute_kernel_hw_startup(cb_x, cb_y);
    cb_wait_front(cb_scaler, 1);
    if constexpr (has_tau) {
        cb_wait_front(cb_tau, 1);
    }
    cb_wait_front(cb_x, n);
    cb_wait_front(cb_t, n);
    for (uint32_t i = 0; i < nitems; ++i) {
        if constexpr (dyn) {
            cb_wait_front(cb_d, nlag);
        }
        tile_regs_acquire();
        for (uint32_t l = 0; l < nlag; ++l) {
            const uint32_t p = i * nlag + l;
            const uint32_t acc = l == 0 ? 0 : 1;
            load(cb_x, p, acc);
            load(cb_t, p, 2);
            if constexpr (dyn) {
                load(cb_d, l, 3);
                binop_with_scalar_tile_init();
                mul_unary_tile(3, cs, R);
                add_binary_tile_init();
                add_binary_tile(2, 3, 2, R);
            }
            mul_binary_tile_init();
            mul_binary_tile(acc, 2, acc, R);
            if (l > 0) {
                add_binary_tile_init();
                add_binary_tile(0, 1, 0, R);
            }
        }
        mul_binary_tile_init();
        mul_binary_tile(0, 0, 1);
        tile_regs_commit();
        tile_regs_wait();
        put(0, cb_y);
        put(1, cb_sq);
        tile_regs_release();
        if constexpr (dyn) {
            cb_pop_front(cb_d, nlag);
        }

        cb_wait_front(cb_sq, 1);
        tile_regs_acquire();
        reconfig_data_format(cb_sq, cb_scaler);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_sq, cb_scaler, cb_f);
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_sq, cb_scaler, 0, 0, 0);
        reduce_uninit();
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv, R);
        if constexpr (norm == 1) {
            add_unary_tile(0, eps, R);
        } else {
            unary_max_tile_init();
            unary_max_tile(0, one);
        }
        rsqrt_tile_init();
        rsqrt_tile(0, R);
        if constexpr (has_tau) {
            load(cb_tau, 0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0, R);
        }
        tile_regs_commit();
        tile_regs_wait();
        put(0, cb_f);
        tile_regs_release();
        cb_pop_front(cb_sq, 1);

        cb_wait_front(cb_y, 1);
        cb_wait_front(cb_f, 1);
        tile_regs_acquire();
        reconfig_data_format(cb_y, cb_f);
        mul_bcast_scalar_init(cb_y, cb_f);
        mul_tiles_bcast_scalar(cb_y, cb_f, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        put(0, cb_out);
        tile_regs_release();
        cb_pop_front(cb_y, 1);
        cb_pop_front(cb_f, 1);
    }
    if constexpr (mass) {
        cb_wait_front(cb_massin, 1);
        tile_regs_acquire();
        load(cb_massin, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        put(0, cb_mass);
        tile_regs_release();
        cb_pop_front(cb_massin, 1);
    }
}
