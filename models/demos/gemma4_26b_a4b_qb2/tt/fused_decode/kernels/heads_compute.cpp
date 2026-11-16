// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-1 q / k / v head norm + RoPE, compute (fused_decode/heads.py: HeadsB1): one packed head tile
// (heads_reader.cpp).
//   rs = rsqrt(sum(x^2) / D + eps)                       (FPU square, scalar reduce, SFPU scalar ops; FP32)
//   v:     out = x * rs
//   q / k: out = ((x * g) * cos + (x_rot * g_rot') * sin) * rs
//          (x_rot: rows rotated by R / 2; g_rot': the gamma rotated the same way with NeoX's sign folded in), i.e.
//          rope(norm(x) * g) with rope(t) = t * cos + rotate_half(t) * sin.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"

constexpr uint32_t kIters = 1;
template <int OP>
ALWI void scalar_op(uint32_t idst, uint32_t s) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, OP, kIters, DST_ACCUM_MODE),
        idst,
        VectorMode::R,
        s));
}

ALWI void pack_one(uint32_t cb) {
    tile_regs_commit();
    cb_reserve_back(cb, 1);
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    pack_tile(0, cb);
    tile_regs_release();
    cb_push_back(cb, 1);
}

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_xr = get_compile_time_arg_val(1);
    constexpr uint32_t cb_cs = get_compile_time_arg_val(2);
    constexpr uint32_t cb_g = get_compile_time_arg_val(3);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(4);
    constexpr uint32_t cb_sq = get_compile_time_arg_val(5);
    constexpr uint32_t cb_rs = get_compile_time_arg_val(6);
    constexpr uint32_t cb_t = get_compile_time_arg_val(7);
    constexpr uint32_t cb_out = get_compile_time_arg_val(8);
    constexpr uint32_t rope = get_compile_time_arg_val(9);
    constexpr uint32_t inv_d = get_compile_time_arg_val(10);  // float bits
    constexpr uint32_t eps = get_compile_time_arg_val(11);

    // rs
    cb_wait_front(cb_x, 1);
    compute_kernel_hw_startup(cb_x, cb_x, cb_sq);
    tile_regs_acquire();
    mul_init(cb_x, cb_x);
    mul_tiles(cb_x, cb_x, 0, 0, 0);
    pack_one(cb_sq);
    cb_wait_front(cb_sq, 1);
    cb_wait_front(cb_scaler, 1);
    compute_kernel_hw_startup(cb_sq, cb_scaler, cb_rs);
    tile_regs_acquire();
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_sq, cb_scaler, cb_rs);
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_sq, cb_scaler, 0, 0, 0);
    reduce_uninit(cb_sq);
    binop_with_scalar_tile_init();
    scalar_op<MUL_UNARY>(0, inv_d);
    scalar_op<ADD_UNARY>(0, eps);
    rsqrt_tile_init();
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_rsqrt, (APPROX, kIters, DST_ACCUM_MODE, false), 0, VectorMode::R));
    pack_one(cb_rs);
    cb_pop_front(cb_sq, 1);
    cb_wait_front(cb_rs, 1);

    uint32_t cb_src = cb_x;
    if constexpr (rope) {
        cb_wait_front(cb_xr, 1);
        cb_wait_front(cb_cs, 2);
        cb_wait_front(cb_g, 2);
        // t1 = (x * g) * cos
        compute_kernel_hw_startup(cb_x, cb_g, cb_t);
        tile_regs_acquire();
        mul_init(cb_x, cb_g);
        mul_tiles(cb_x, cb_g, 0, 0, 0);
        mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_cs);
        mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_cs, 0, 0);
        pack_one(cb_t);
        // t2 = (x_rot * g_rot') * sin + t1
        compute_kernel_hw_startup(cb_xr, cb_g, cb_t);
        tile_regs_acquire();
        mul_init(cb_xr, cb_g);
        mul_tiles(cb_xr, cb_g, 0, 1, 0);
        mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_cs);
        mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_cs, 1, 0);
        cb_wait_front(cb_t, 1);
        reconfig_data_format_srcb(cb_cs, cb_t);
        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_t);
        add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_t, 0, 0);
        pack_one(cb_t);
        cb_pop_front(cb_t, 1);
        cb_wait_front(cb_t, 1);
        cb_src = cb_t;
    }
    // out = src * rs
    compute_kernel_hw_startup(cb_src, cb_rs, cb_out);
    tile_regs_acquire();
    mul_bcast_scalar_init(cb_src, cb_rs);
    mul_tiles_bcast_scalar(cb_src, cb_rs, 0, 0, 0);
    pack_one(cb_out);
}
