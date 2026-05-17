// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// SigLIP row-wise softmax kernel — numerically stable, per-row max-subtract.
//
// Math (per row r):
//   m_r        = max_k x[r, k]
//   xms[r, k]  = x[r, k] - m_r
//   exp[r, k]  = exp(xms[r, k])
//   s_r        = sum_k exp[r, k]
//   out[r, k]  = exp[r, k] / s_r
//
// Decomposition: 8 cores × 1 M-tile (32 rows) per core. K=1152 → 36 K-tiles.
//
// Pattern: copies softmax_sharded.cpp's `calc_numeric_stable` loop semantics
// (multi-call reduce_tile WITHIN a tile_regs_acquire is the canonical pattern).

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
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/tile_move_copy.h"
#endif

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t in_cb = get_named_compile_time_arg_val("in_cb");
    constexpr uint32_t scaler_cb = get_named_compile_time_arg_val("scaler_cb");
    constexpr uint32_t out_cb = get_named_compile_time_arg_val("out_cb");
    constexpr uint32_t in_tiles = get_named_compile_time_arg_val("in_tiles");

    unified_kernels::setup_sharded_buffer(in_cb, in_tiles);
    unified_kernels::setup_sharded_buffer(scaler_cb, 1);
    unified_kernels::setup_sharded_buffer(out_cb, in_tiles);
    DPRINT << "NC: setup. in_tiles=" << in_tiles << ENDL();

#elif defined(COMPILE_FOR_BRISC)
    // no-op

#elif defined(COMPILE_FOR_TRISC)
    constexpr uint32_t in_cb = get_named_compile_time_arg_val("in_cb");
    constexpr uint32_t scaler_cb = get_named_compile_time_arg_val("scaler_cb");
    constexpr uint32_t max_cb = get_named_compile_time_arg_val("max_cb");
    constexpr uint32_t exp_cb = get_named_compile_time_arg_val("exp_cb");
    constexpr uint32_t sum_cb = get_named_compile_time_arg_val("sum_cb");
    constexpr uint32_t isum_cb = get_named_compile_time_arg_val("isum_cb");
    constexpr uint32_t out_cb = get_named_compile_time_arg_val("out_cb");
    constexpr uint32_t K_TILES = get_named_compile_time_arg_val("k_tiles");    // 36
    constexpr uint32_t M_TILES = get_named_compile_time_arg_val("m_tiles");    // 1 per core
    constexpr uint32_t IN_TILES = get_named_compile_time_arg_val("in_tiles");  // M_TILES*K_TILES = 36

    cb_wait_front(in_cb, IN_TILES);
    cb_wait_front(scaler_cb, 1);

    // One-time LLK state init for binary compute (mul/sub/add tiles family).
    // See [[pi05-llk-binary-op-init-common]] memory — required for any custom
    // TRISC kernel using binary tile ops, else TR0/TR1 hang.
    binary_op_init_common(in_cb, scaler_cb, max_cb);

    DPRINT << "TR: ready. K_TILES=" << K_TILES << " M_TILES=" << M_TILES << ENDL();

    // Process each M-tile (here M_TILES=1, but the loop generalizes).
    for (uint32_t m = 0; m < M_TILES; ++m) {
        uint32_t in_row_offset = m * K_TILES;

        // ============================================================
        // PHASE 1: row max via reduce_tile<MAX, ROW> looped over K_TILES.
        // ============================================================
        reconfig_data_format(in_cb, scaler_cb);
        pack_reconfig_data_format(max_cb);

        cb_reserve_back(max_cb, 1);
        tile_regs_acquire();
        reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW, /*enforce_fp32=*/true>(in_cb, scaler_cb, max_cb);
        for (uint32_t k = 0; k < K_TILES; ++k) {
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW, /*enforce_fp32=*/true>(
                in_cb, scaler_cb, in_row_offset + k, 0, 0);
        }
        reduce_uninit();
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, max_cb);
        tile_regs_release();
        cb_push_back(max_cb, 1);
        DPRINT << "TR: P1 max done m=" << m << ENDL();

        // ============================================================
        // PHASE 2: exp(x - max) for each K-tile in this row.
        // ============================================================
        cb_wait_front(max_cb, 1);
        reconfig_data_format(in_cb, max_cb);
        pack_reconfig_data_format(exp_cb);
        sub_bcast_cols_init_short(in_cb, max_cb);
        exp_tile_init();

        cb_reserve_back(exp_cb, K_TILES);
        for (uint32_t k = 0; k < K_TILES; ++k) {
            tile_regs_acquire();
            sub_tiles_bcast<BroadcastType::COL>(in_cb, max_cb, in_row_offset + k, 0, 0);
            exp_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, exp_cb);
            tile_regs_release();
        }
        cb_push_back(exp_cb, K_TILES);
        cb_pop_front(max_cb, 1);
        DPRINT << "TR: P2 exp done m=" << m << ENDL();

        // ============================================================
        // PHASE 3: row sum via reduce_tile<SUM, ROW> looped over K_TILES.
        // ============================================================
        cb_wait_front(exp_cb, K_TILES);
        reconfig_data_format(exp_cb, scaler_cb);
        pack_reconfig_data_format(sum_cb);

        cb_reserve_back(sum_cb, 1);
        tile_regs_acquire();
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW, /*enforce_fp32=*/true>(exp_cb, scaler_cb, sum_cb);
        for (uint32_t k = 0; k < K_TILES; ++k) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW, /*enforce_fp32=*/true>(exp_cb, scaler_cb, k, 0, 0);
        }
        reduce_uninit();
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, sum_cb);
        tile_regs_release();
        cb_push_back(sum_cb, 1);
        DPRINT << "TR: P3 sum done m=" << m << ENDL();

        // ============================================================
        // PHASE 4: isum = 1 / sum   →  isum_cb (single tile).
        // ============================================================
        cb_wait_front(sum_cb, 1);
        reconfig_data_format_srca(sum_cb);
        pack_reconfig_data_format(isum_cb);
        copy_tile_to_dst_init_short(sum_cb);
        recip_tile_init();

        cb_reserve_back(isum_cb, 1);
        tile_regs_acquire();
        copy_tile(sum_cb, 0, 0);
        recip_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, isum_cb);
        tile_regs_release();
        cb_push_back(isum_cb, 1);
        cb_pop_front(sum_cb, 1);
        DPRINT << "TR: P4 isum done m=" << m << ENDL();

        // ============================================================
        // PHASE 5: out = exp * isum  (broadcast COL).
        // ============================================================
        cb_wait_front(isum_cb, 1);
        reconfig_data_format(exp_cb, isum_cb);
        pack_reconfig_data_format(out_cb);
        mul_bcast_cols_init_short(exp_cb, isum_cb);

        cb_reserve_back(out_cb, K_TILES);
        for (uint32_t k = 0; k < K_TILES; ++k) {
            tile_regs_acquire();
            mul_tiles_bcast<BroadcastType::COL>(exp_cb, isum_cb, k, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, out_cb);
            tile_regs_release();
        }
        cb_push_back(out_cb, K_TILES);
        cb_pop_front(exp_cb, K_TILES);
        cb_pop_front(isum_cb, 1);
        DPRINT << "TR: P5 out done m=" << m << " — finished." << ENDL();
    }
#endif
}
