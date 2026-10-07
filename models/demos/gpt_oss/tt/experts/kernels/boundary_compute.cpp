// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode layer boundary, compute (TRISC) (tt/decode_boundary.py: DecodeBoundary).
//
// The decode residual stream is a flat BF16 vector stored in `tiles` 32x32 tile pages (hidden value h at byte 2 h,
// zero padded). Every op here is elementwise or a whole-vector sum, so the tile structure is only a container:
//   1. residual' = (sum of the `slots` partial sums in slot order) + residual: the same order on every device of
//      the TP ring (bit-identical replicated residual), the partials first so the BF16 rounding of the running sum
//      happens at their magnitude; packed into cb_res_out (backed by the output residual);
//   2. ss = sum(residual'^2) (FP32, squares of the BF16-rounded residual, as the HF RMSNorm sees it);
//   3. x = (residual' * gamma) * rsqrt(ss / hidden + eps), packed into cb_x (backed by the normed output).
// All tile loads go through the FPU (unpack A/B); the SFPU only works on DST. The FPU phases reading the FP32
// intermediates (the scalar reduce, the scalar broadcast) start with a full compute_kernel_hw_startup (as groupnorm
// does mid kernel): with only the op inits the reduce sums part of the tile. BND_* zones: device-profiler breakdown
// (compiled out without the profiler).

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "tools/profiler/kernel_profiler.hpp"

// SFPU ops on the scalar statistic in element (0, 0) of DST: first rows of faces 0 / 1 only (as the gate|up stream's
// row ops), not the whole tile.
constexpr uint32_t kScalarIters = 1;
template <int OP>
ALWI void scalar_op(uint32_t idst, uint32_t s) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_binop_with_scalar,
        (APPROX, OP, kScalarIters, DST_ACCUM_MODE),
        idst,
        VectorMode::R,
        s));
}
ALWI void rsqrt_scalar(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_rsqrt,
        (APPROX, kScalarIters, DST_ACCUM_MODE, false),
        idst,
        VectorMode::R));
}

void kernel_main() {
    constexpr uint32_t cb_res = get_compile_time_arg_val(0);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(3);
    constexpr uint32_t cb_res_out = get_compile_time_arg_val(4);
    constexpr uint32_t cb_sq = get_compile_time_arg_val(5);
    constexpr uint32_t cb_rs = get_compile_time_arg_val(6);
    constexpr uint32_t cb_x = get_compile_time_arg_val(7);
    constexpr uint32_t tiles = get_compile_time_arg_val(8);
    constexpr uint32_t slots = get_compile_time_arg_val(9);
    constexpr uint32_t inv_n = get_compile_time_arg_val(10);  // float bits of 1 / hidden
    constexpr uint32_t eps = get_compile_time_arg_val(11);    // float bits
    constexpr uint32_t do_norm = get_compile_time_arg_val(12);
    constexpr uint32_t cb_y = get_compile_time_arg_val(13);
    static_assert(tiles <= 3, "phase 2 / 3 hold the tiles plus one operand in the 4 FP32 DST tiles");

    // 1. residual' = ((slot 0 + slot 1) + ...) + residual: the partial sums first (rounded at their own magnitude
    //    when DST feeds SrcA), the residual last (FPU, FP32 DST).
    compute_kernel_hw_startup(slots > 0 ? cb_recv : cb_res, cb_res, cb_res_out);
    cb_wait_front(cb_res, tiles);
    if constexpr (slots > 0) {
        cb_wait_front(cb_recv, slots * tiles);
    }
    cb_reserve_back(cb_res_out, tiles);
    for (uint32_t t = 0; t < tiles; ++t) {
        DeviceZoneScopedN("BND_C1_SUM");
        tile_regs_acquire();
        if constexpr (slots > 1) {
            add_init(cb_recv, cb_recv);
            add_tiles(cb_recv, cb_recv, t, tiles + t, 0);
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_recv);
            for (uint32_t s = 2; s < slots; ++s) {
                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_recv, s * tiles + t, 0);
            }
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_res);
            add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_res, t, 0);
        } else if constexpr (slots == 1) {
            add_init(cb_recv, cb_res);
            add_tiles(cb_recv, cb_res, t, t, 0);
        } else {
            copy_init(cb_res);
            copy_tile(cb_res, t, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_res_out);
        tile_regs_release();
    }
    cb_push_back(cb_res_out, tiles);
    cb_pop_front(cb_res, tiles);
    if constexpr (slots > 0) {
        cb_pop_front(cb_recv, slots * tiles);
    }
    if constexpr (!do_norm) {
        return;
    }

    // 2. y = residual' * gamma (FPU, FP32, packed to cb_y) and the per-position sum over the tiles of residual'^2
    //    (FPU squares of the BF16-rounded residual, summed in DST by the SFPU, packed to cb_sq).
    cb_wait_front(cb_res_out, tiles);
    cb_wait_front(cb_gamma, tiles);
    {
        DeviceZoneScopedN("BND_C2_SQ");
        tile_regs_acquire();
        mul_init(cb_res_out, cb_gamma);
        for (uint32_t t = 0; t < tiles; ++t) {
            mul_tiles(cb_res_out, cb_gamma, t, t, t);
        }
        tile_regs_commit();
        cb_reserve_back(cb_y, tiles);
        tile_regs_wait();
        pack_reconfig_data_format(cb_y);
        for (uint32_t t = 0; t < tiles; ++t) {
            pack_tile(t, cb_y);
        }
        tile_regs_release();
        cb_push_back(cb_y, tiles);

        tile_regs_acquire();
        mul_init(cb_res_out, cb_res_out);
        for (uint32_t t = 0; t < tiles; ++t) {
            mul_tiles(cb_res_out, cb_res_out, t, t, t);
        }
        add_binary_tile_init();
        for (uint32_t t = 1; t < tiles; ++t) {
            add_binary_tile(0, t, 0);
        }
        tile_regs_commit();
        cb_reserve_back(cb_sq, 1);
        tile_regs_wait();
        pack_tile(0, cb_sq);
        tile_regs_release();
        cb_push_back(cb_sq, 1);
    }
    cb_pop_front(cb_gamma, tiles);

    // 3. rs = rsqrt(sum / hidden + eps): FPU scalar reduce, SFPU scalar ops.
    cb_wait_front(cb_sq, 1);
    cb_wait_front(cb_scaler, 1);
    {
        DeviceZoneScopedN("BND_C3_RS");
        compute_kernel_hw_startup(cb_sq, cb_scaler, cb_rs);
        tile_regs_acquire();
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_sq, cb_scaler, cb_rs);
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_sq, cb_scaler, 0, 0, 0);
        reduce_uninit(cb_sq);
        binop_with_scalar_tile_init();
        scalar_op<MUL_UNARY>(0, inv_n);
        scalar_op<ADD_UNARY>(0, eps);
        rsqrt_tile_init();
        rsqrt_scalar(0);
        tile_regs_commit();
        cb_reserve_back(cb_rs, 1);
        tile_regs_wait();
        pack_tile(0, cb_rs);
        tile_regs_release();
        cb_push_back(cb_rs, 1);
    }
    cb_pop_front(cb_sq, 1);
    cb_pop_front(cb_scaler, 1);

    // 4. x = y * rs (FPU scalar broadcast).
    cb_wait_front(cb_rs, 1);
    cb_wait_front(cb_y, tiles);
    cb_reserve_back(cb_x, tiles);
    {
        DeviceZoneScopedN("BND_C4_X");
        compute_kernel_hw_startup(cb_y, cb_rs, cb_x);
        tile_regs_acquire();
        mul_bcast_scalar_init(cb_y, cb_rs);
        for (uint32_t t = 0; t < tiles; ++t) {
            mul_tiles_bcast_scalar(cb_y, cb_rs, t, 0, t);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_x);
        for (uint32_t t = 0; t < tiles; ++t) {
            pack_tile(t, cb_x);
        }
        tile_regs_release();
    }
    cb_push_back(cb_x, tiles);
    cb_pop_front(cb_y, tiles);
    cb_pop_front(cb_rs, 1);
}
