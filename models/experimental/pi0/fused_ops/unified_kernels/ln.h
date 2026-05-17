// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// SigLIP LayerNorm Op-struct.
//
// Math (per row r in this core's M-tile):
//   mean[r] = sum_d(x[r,d]) / D
//   var[r]  = sum_d((x[r,d] - mean[r])^2) / D
//   y[r,d]  = ((x[r,d] - mean[r]) / sqrt(var[r] + eps)) * gamma[d] + beta[d]
//
// Ported from siglip_layernorm_kernel.cpp (v3). Preserves the Stage-A
// (mul_tiles accumulate) + Stage-B (single reduce_tile) pattern in Phases 1
// and 4, plus the mandatory binary_op_init_common reset between phase types
// — both load-bearing per pi05-ln-kernel-multi-reduce-pattern and
// pi05-llk-binary-op-init-common memories.
//
// Decomposition: 8 cores × 1 M-tile (32 rows) per core. D_TILES = 36.

#pragma once

// REDUCE_OP / REDUCE_DIM must be defined before reduce.h is included. Done
// here (not in the kernel main) so ln.h is self-contained. Note: setting
// these via #define is a TU-wide effect — any other Op-struct header included
// in the same TU that needs different reduce config must guard with
// #ifndef / #define and reset, or be included BEFORE ln.h.
#ifndef REDUCE_OP
#define REDUCE_OP PoolType::SUM
#endif
#ifndef REDUCE_DIM
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#endif

#include "../../../../demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

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

namespace pi05_siglip_ops {

struct LayerNorm {
    struct ReaderCTArgs {};
    struct WriterCTArgs {};

    template <
        uint32_t InCb,
        uint32_t GammaCb,
        uint32_t BetaCb,
        uint32_t ScalerCb,
        uint32_t OnesCb,
        uint32_t AccumCb,
        uint32_t XmmCb,
        uint32_t Xmm2Cb,
        uint32_t MeanCb,
        uint32_t VarCb,
        uint32_t IvarCb,
        uint32_t OutCb,
        uint32_t DTiles,
        uint32_t InTiles,
        uint32_t EpsBits>
    struct ComputeCTArgs {
        static constexpr uint32_t in_cb = InCb;
        static constexpr uint32_t gamma_cb = GammaCb;
        static constexpr uint32_t beta_cb = BetaCb;
        static constexpr uint32_t scaler_cb = ScalerCb;
        static constexpr uint32_t ones_cb = OnesCb;
        static constexpr uint32_t accum_cb = AccumCb;
        static constexpr uint32_t xmm_cb = XmmCb;
        static constexpr uint32_t xmm2_cb = Xmm2Cb;
        static constexpr uint32_t mean_cb = MeanCb;
        static constexpr uint32_t var_cb = VarCb;
        static constexpr uint32_t ivar_cb = IvarCb;
        static constexpr uint32_t out_cb = OutCb;
        static constexpr uint32_t d_tiles = DTiles;
        static constexpr uint32_t in_tiles = InTiles;
        static constexpr uint32_t eps_bits = EpsBits;
    };

    struct ReaderArgs {};
    struct WriterArgs {};
    struct ComputeArgs {};

    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

    template <typename CTArgs, bool IsActiveCore>
    class Op {
    public:
        void operator()(const RTArgs& args) {
            if constexpr (IsActiveCore) {
                impl(args);
            }
        }

    private:
        void impl(const RTArgs& /*args*/) {
#if defined(COMPILE_FOR_TRISC)
            constexpr uint32_t in_cb = CTArgs::in_cb;
            constexpr uint32_t gamma_cb = CTArgs::gamma_cb;
            constexpr uint32_t beta_cb = CTArgs::beta_cb;
            constexpr uint32_t scaler_cb = CTArgs::scaler_cb;
            constexpr uint32_t ones_cb = CTArgs::ones_cb;
            constexpr uint32_t accum_cb = CTArgs::accum_cb;
            constexpr uint32_t xmm_cb = CTArgs::xmm_cb;
            constexpr uint32_t xmm2_cb = CTArgs::xmm2_cb;
            constexpr uint32_t mean_cb = CTArgs::mean_cb;
            constexpr uint32_t var_cb = CTArgs::var_cb;
            constexpr uint32_t ivar_cb = CTArgs::ivar_cb;
            constexpr uint32_t out_cb = CTArgs::out_cb;
            constexpr uint32_t D_TILES = CTArgs::d_tiles;
            constexpr uint32_t IN_TILES = CTArgs::in_tiles;
            constexpr uint32_t eps_bits = CTArgs::eps_bits;

            cb_wait_front(in_cb, IN_TILES);
            cb_wait_front(gamma_cb, D_TILES);
            cb_wait_front(beta_cb, D_TILES);
            cb_wait_front(scaler_cb, 1);
            cb_wait_front(ones_cb, 1);

            // Mandatory binary-op LLK init (pi05-llk-binary-op-init-common):
            // without it, UNPACK/MATH hang while PACK proceeds (the
            // "TR2 prints, TR0/TR1 hang" deadlock).
            binary_op_init_common(in_cb, ones_cb, accum_cb);

            // ============================================================
            // PHASE 1: row mean — Stage A (mul accumulate) + Stage B (reduce ROW).
            // ============================================================
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

            // Stage B: must reset UNPACK state via binary_op_init_common
            // after mul_tiles_init, BEFORE reduce_init. Without this, the
            // single reduce_tile call hangs (groupnorm reference pattern).
            cb_wait_front(accum_cb, 1);
            binary_op_init_common(accum_cb, scaler_cb, mean_cb);
            tile_regs_acquire();
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW, /*enforce_fp32_accumulation=*/true>(
                accum_cb, scaler_cb, mean_cb);
            cb_reserve_back(mean_cb, 1);
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW, /*enforce_fp32_accumulation=*/true>(
                accum_cb, scaler_cb, 0, 0, 0);
            cb_pop_front(accum_cb, 1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, mean_cb);
            tile_regs_release();
            cb_push_back(mean_cb, 1);
            reduce_uninit();

            // ============================================================
            // PHASE 2: xmm = x - mean   (broadcast COL across 32 cols).
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

            // ============================================================
            // PHASE 3: xmm2 = xmm * xmm   (elementwise square per tile).
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

            // ============================================================
            // PHASE 4: row variance — Stage A + Stage B (same reset pattern).
            // ============================================================
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

            cb_wait_front(accum_cb, 1);
            binary_op_init_common(accum_cb, scaler_cb, var_cb);
            tile_regs_acquire();
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW, /*enforce_fp32_accumulation=*/true>(
                accum_cb, scaler_cb, var_cb);
            cb_reserve_back(var_cb, 1);
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW, /*enforce_fp32_accumulation=*/true>(
                accum_cb, scaler_cb, 0, 0, 0);
            cb_pop_front(accum_cb, 1);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, var_cb);
            tile_regs_release();
            cb_push_back(var_cb, 1);
            reduce_uninit();

            // ============================================================
            // PHASE 5: ivar = 1 / sqrt(var + eps).
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

            // ============================================================
            // PHASE 6: normalized = xmm * ivar (broadcast COL) → xmm2_cb.
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

            // ============================================================
            // PHASE 7: scaled = normalized * gamma (broadcast ROW) → xmm_cb.
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

            // ============================================================
            // PHASE 8: out = scaled + beta (broadcast ROW).
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
#endif
        }
    };
};

}  // namespace pi05_siglip_ops
