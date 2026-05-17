// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// SigLIP LayerNorm kernel — per-row LN with scale (gamma) + bias (beta).
//
// Math (per row r in this core's M-tile):
//   mean[r] = sum_d(x[r,d]) / D
//   var[r]  = sum_d((x[r,d] - mean[r])^2) / D
//   y[r,d]  = ((x[r,d] - mean[r]) / sqrt(var[r] + eps)) * gamma[d] + beta[d]
//
// Decomposition: 8 cores × 1 M-tile (32 rows) per core. D_TILES = 36.
//
// **v3 (Stage-A/Stage-B reduce pattern):** Multi-call reduce_tile accumulation
// hangs the LLK (see [[pi05-ln-kernel-multi-reduce-pattern]]). Phases 1 and 4
// use:
//   Stage A: loop mul_tiles(in_cb, ones_cb, k, 0, dst0)  ← accumulates into dst0
//   Stage B: single reduce_tile<SUM, ROW>(accumulator, scaler, 0, 0, dst0)
// pattern (per groupnorm_sharded_v2.cpp:240-278 reference).

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "../../../../demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"
#include "../../../../demos/deepseek_v3_b1/unified_kernels/kernel_utils.hpp"
#include "api/debug/dprint.h"

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "../../../../demos/deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/add_rsqrt.h"
#endif

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t in_cb = get_named_compile_time_arg_val("in_cb");
    constexpr uint32_t gamma_cb = get_named_compile_time_arg_val("gamma_cb");
    constexpr uint32_t beta_cb = get_named_compile_time_arg_val("beta_cb");
    constexpr uint32_t scaler_cb = get_named_compile_time_arg_val("scaler_cb");
    constexpr uint32_t ones_cb = get_named_compile_time_arg_val("ones_cb");
    constexpr uint32_t in_tiles = get_named_compile_time_arg_val("in_tiles");
    constexpr uint32_t gamma_tiles = get_named_compile_time_arg_val("gamma_tiles");

    unified_kernels::setup_sharded_buffer(in_cb, in_tiles);
    unified_kernels::setup_sharded_buffer(gamma_cb, gamma_tiles);
    unified_kernels::setup_sharded_buffer(beta_cb, gamma_tiles);
    unified_kernels::setup_sharded_buffer(scaler_cb, 1);
    unified_kernels::setup_sharded_buffer(ones_cb, 1);
    DPRINT << "NC: setup. in_tiles=" << in_tiles << " gamma_tiles=" << gamma_tiles << ENDL();

#elif defined(COMPILE_FOR_BRISC)
    // no-op

#elif defined(COMPILE_FOR_TRISC)
    constexpr uint32_t in_cb = get_named_compile_time_arg_val("in_cb");
    constexpr uint32_t gamma_cb = get_named_compile_time_arg_val("gamma_cb");
    constexpr uint32_t beta_cb = get_named_compile_time_arg_val("beta_cb");
    constexpr uint32_t scaler_cb = get_named_compile_time_arg_val("scaler_cb");
    constexpr uint32_t ones_cb = get_named_compile_time_arg_val("ones_cb");
    constexpr uint32_t accum_cb = get_named_compile_time_arg_val("accum_cb");
    constexpr uint32_t xmm_cb = get_named_compile_time_arg_val("xmm_cb");
    constexpr uint32_t xmm2_cb = get_named_compile_time_arg_val("xmm2_cb");
    constexpr uint32_t mean_cb = get_named_compile_time_arg_val("mean_cb");
    constexpr uint32_t var_cb = get_named_compile_time_arg_val("var_cb");
    constexpr uint32_t ivar_cb = get_named_compile_time_arg_val("ivar_cb");
    constexpr uint32_t out_cb = get_named_compile_time_arg_val("out_cb");
    constexpr uint32_t D_TILES = get_named_compile_time_arg_val("d_tiles");
    constexpr uint32_t IN_TILES = get_named_compile_time_arg_val("in_tiles");
    constexpr uint32_t eps_bits = get_named_compile_time_arg_val("eps_bits");

    cb_wait_front(in_cb, IN_TILES);
    cb_wait_front(gamma_cb, D_TILES);
    cb_wait_front(beta_cb, D_TILES);
    cb_wait_front(scaler_cb, 1);
    cb_wait_front(ones_cb, 1);

    // One-time LLK state init for binary compute (mul/add/sub_tiles family).
    // Required before any binary *_tiles_init call — without it UNPACK and MATH
    // never finish their first op (only PACK proceeds, giving the classic
    // "TR2 prints, TR0/TR1 hang" deadlock). See
    // ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_kernel.cpp:33.
    binary_op_init_common(in_cb, ones_cb, accum_cb);

    DPRINT << "TR: ready. D_TILES=" << D_TILES << " IN_TILES=" << IN_TILES << ENDL();

    // ============================================================
    // PHASE 1: row mean — Stage A (mul accumulate) + Stage B (reduce ROW).
    // ============================================================
    // Stage A: accumulate column-aligned per-tile sums into dst[0] via
    //          mul_tiles(in, ones, k, 0, dst0). Pack accumulator (32, 32)
    //          → accum_cb (1 tile).
    reconfig_data_format(in_cb, ones_cb);
    pack_reconfig_data_format(accum_cb);
    mul_tiles_init(in_cb, ones_cb);

    cb_reserve_back(accum_cb, 1);
    tile_regs_acquire();
    for (uint32_t k = 0; k < D_TILES; ++k) {
        mul_tiles(in_cb, ones_cb, k, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, accum_cb);
    tile_regs_release();
    cb_push_back(accum_cb, 1);
    DPRINT << "TR: P1.A accum done" << ENDL();

    // Stage B (groupnorm-ordered). HYPOTHESIS 1: reset UNPACK state via
    // binary_op_init_common after mul_tiles_init/mul_tiles, BEFORE reduce_init.
    // Stock LN kernel calls binary_op_init_common to reset state between op types.
    cb_wait_front(accum_cb, 1);
    binary_op_init_common(accum_cb, scaler_cb, mean_cb);
    DPRINT << "TR: P1.B binary_op_init_common done" << ENDL();
    tile_regs_acquire();
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW, /*enforce_fp32_accumulation=*/true>(accum_cb, scaler_cb, mean_cb);
    cb_reserve_back(mean_cb, 1);
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW, /*enforce_fp32_accumulation=*/true>(accum_cb, scaler_cb, 0, 0, 0);
    cb_pop_front(accum_cb, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, mean_cb);
    tile_regs_release();
    cb_push_back(mean_cb, 1);
    reduce_uninit();
    DPRINT << "TR: P1.B mean done" << ENDL();

    // ============================================================
    // PHASE 2: xmm = x - mean   (broadcast COL across 32 cols of each tile)
    // ============================================================
    cb_wait_front(mean_cb, 1);
    reconfig_data_format(in_cb, mean_cb);
    pack_reconfig_data_format(xmm_cb);
    sub_bcast_cols_init_short(in_cb, mean_cb);

    cb_reserve_back(xmm_cb, D_TILES);
    for (uint32_t k = 0; k < D_TILES; ++k) {
        tile_regs_acquire();
        sub_tiles_bcast<BroadcastType::COL>(in_cb, mean_cb, k, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, xmm_cb);
        tile_regs_release();
    }
    cb_push_back(xmm_cb, D_TILES);
    cb_pop_front(mean_cb, 1);
    cb_pop_front(in_cb, IN_TILES);
    DPRINT << "TR: P2 xmm done" << ENDL();

    // ============================================================
    // PHASE 3: xmm2 = xmm * xmm   (elementwise square per tile)
    // ============================================================
    cb_wait_front(xmm_cb, D_TILES);
    reconfig_data_format_srca(xmm_cb);
    pack_reconfig_data_format(xmm2_cb);
    copy_tile_to_dst_init_short(xmm_cb);
    square_tile_init();

    cb_reserve_back(xmm2_cb, D_TILES);
    for (uint32_t k = 0; k < D_TILES; ++k) {
        tile_regs_acquire();
        copy_tile(xmm_cb, k, 0);
        square_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, xmm2_cb);
        tile_regs_release();
    }
    cb_push_back(xmm2_cb, D_TILES);
    DPRINT << "TR: P3 xmm2 done" << ENDL();

    // ============================================================
    // PHASE 4: row variance — Stage A (mul accumulate) + Stage B (reduce ROW).
    // ============================================================
    // Stage A: dst[0] = sum_k(xmm2[k] * 1.0)
    cb_wait_front(xmm2_cb, D_TILES);
    reconfig_data_format(xmm2_cb, ones_cb);
    pack_reconfig_data_format(accum_cb);
    mul_tiles_init(xmm2_cb, ones_cb);

    cb_reserve_back(accum_cb, 1);
    tile_regs_acquire();
    for (uint32_t k = 0; k < D_TILES; ++k) {
        mul_tiles(xmm2_cb, ones_cb, k, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, accum_cb);
    tile_regs_release();
    cb_push_back(accum_cb, 1);
    cb_pop_front(xmm2_cb, D_TILES);
    DPRINT << "TR: P4.A accum done" << ENDL();

    // Stage B (same hypothesis-1 reset as Phase 1 Stage B).
    cb_wait_front(accum_cb, 1);
    binary_op_init_common(accum_cb, scaler_cb, var_cb);
    tile_regs_acquire();
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW, /*enforce_fp32_accumulation=*/true>(accum_cb, scaler_cb, var_cb);
    cb_reserve_back(var_cb, 1);
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW, /*enforce_fp32_accumulation=*/true>(accum_cb, scaler_cb, 0, 0, 0);
    cb_pop_front(accum_cb, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, var_cb);
    tile_regs_release();
    cb_push_back(var_cb, 1);
    reduce_uninit();
    DPRINT << "TR: P4.B var done" << ENDL();

    // ============================================================
    // PHASE 5: ivar = 1 / sqrt(var + eps)   →  ivar_cb
    // ============================================================
    cb_wait_front(var_cb, 1);
    reconfig_data_format_srca(var_cb);
    pack_reconfig_data_format(ivar_cb);
    copy_tile_to_dst_init_short(var_cb);
    add_rsqrt_tile_init();

    cb_reserve_back(ivar_cb, 1);
    tile_regs_acquire();
    copy_tile(var_cb, 0, 0);
    add_rsqrt_tile(0, eps_bits);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, ivar_cb);
    tile_regs_release();
    cb_push_back(ivar_cb, 1);
    cb_pop_front(var_cb, 1);
    DPRINT << "TR: P5 ivar done" << ENDL();

    // ============================================================
    // PHASE 6: normalized = xmm * ivar   (broadcast COL)   → xmm2_cb (recycled)
    // ============================================================
    cb_wait_front(ivar_cb, 1);
    reconfig_data_format(xmm_cb, ivar_cb);
    pack_reconfig_data_format(xmm2_cb);
    mul_bcast_cols_init_short(xmm_cb, ivar_cb);

    cb_reserve_back(xmm2_cb, D_TILES);
    for (uint32_t k = 0; k < D_TILES; ++k) {
        tile_regs_acquire();
        mul_tiles_bcast<BroadcastType::COL>(xmm_cb, ivar_cb, k, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, xmm2_cb);
        tile_regs_release();
    }
    cb_push_back(xmm2_cb, D_TILES);
    cb_pop_front(xmm_cb, D_TILES);
    cb_pop_front(ivar_cb, 1);
    DPRINT << "TR: P6 normalize done" << ENDL();

    // ============================================================
    // PHASE 7: scaled = normalized * gamma   (broadcast ROW)   → xmm_cb (recycled)
    // ============================================================
    cb_wait_front(xmm2_cb, D_TILES);
    reconfig_data_format(xmm2_cb, gamma_cb);
    pack_reconfig_data_format(xmm_cb);
    mul_bcast_rows_init_short(xmm2_cb, gamma_cb);

    cb_reserve_back(xmm_cb, D_TILES);
    for (uint32_t k = 0; k < D_TILES; ++k) {
        tile_regs_acquire();
        mul_tiles_bcast<BroadcastType::ROW>(xmm2_cb, gamma_cb, k, k, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, xmm_cb);
        tile_regs_release();
    }
    cb_push_back(xmm_cb, D_TILES);
    cb_pop_front(xmm2_cb, D_TILES);
    DPRINT << "TR: P7 gamma done" << ENDL();

    // ============================================================
    // PHASE 8: out = scaled + beta   (broadcast ROW)
    // ============================================================
    cb_wait_front(xmm_cb, D_TILES);
    reconfig_data_format(xmm_cb, beta_cb);
    pack_reconfig_data_format(out_cb);
    add_bcast_rows_init_short(xmm_cb, beta_cb);

    cb_reserve_back(out_cb, D_TILES);
    for (uint32_t k = 0; k < D_TILES; ++k) {
        tile_regs_acquire();
        add_tiles_bcast<BroadcastType::ROW>(xmm_cb, beta_cb, k, k, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
    }
    cb_push_back(out_cb, D_TILES);
    cb_pop_front(xmm_cb, D_TILES);
    DPRINT << "TR: P8 beta done — finished" << ENDL();
#endif
}
