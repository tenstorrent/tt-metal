// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// SigLIP row-wise softmax Op-struct.
//
// Math (per row r):
//   m_r        = max_k x[r, k]
//   xms[r, k]  = x[r, k] - m_r
//   exp[r, k]  = exp(xms[r, k])
//   s_r        = sum_k exp[r, k]
//   out[r, k]  = exp[r, k] / s_r
//
// Ported from siglip_softmax_kernel.cpp. Used inside SDPA per head between
// QK^T and Attn@V matmuls. Numerically stable max-subtract.
//
// Decomposition: per-row, M-parallel sharded. Default for SDPA: 8 cores
// × 1 M-tile (32 rows) per core. K_TILES varies by SDPA shape.

#pragma once

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
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/tile_move_copy.h"
#endif

namespace pi05_siglip_ops {

struct Softmax {
    struct ReaderCTArgs {};
    struct WriterCTArgs {};

    template <
        uint32_t InCb,
        uint32_t ScalerCb,
        uint32_t MaxCb,
        uint32_t ExpCb,
        uint32_t SumCb,
        uint32_t IsumCb,
        uint32_t OutCb,
        uint32_t KTiles,
        uint32_t MTiles,
        uint32_t InTiles>
    struct ComputeCTArgs {
        static constexpr uint32_t in_cb = InCb;
        static constexpr uint32_t scaler_cb = ScalerCb;
        static constexpr uint32_t max_cb = MaxCb;
        static constexpr uint32_t exp_cb = ExpCb;
        static constexpr uint32_t sum_cb = SumCb;
        static constexpr uint32_t isum_cb = IsumCb;
        static constexpr uint32_t out_cb = OutCb;
        static constexpr uint32_t k_tiles = KTiles;
        static constexpr uint32_t m_tiles = MTiles;
        static constexpr uint32_t in_tiles = InTiles;
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
            constexpr uint32_t scaler_cb = CTArgs::scaler_cb;
            constexpr uint32_t max_cb = CTArgs::max_cb;
            constexpr uint32_t exp_cb = CTArgs::exp_cb;
            constexpr uint32_t sum_cb = CTArgs::sum_cb;
            constexpr uint32_t isum_cb = CTArgs::isum_cb;
            constexpr uint32_t out_cb = CTArgs::out_cb;
            constexpr uint32_t K_TILES = CTArgs::k_tiles;
            constexpr uint32_t M_TILES = CTArgs::m_tiles;
            constexpr uint32_t IN_TILES = CTArgs::in_tiles;

            cb_wait_front(in_cb, IN_TILES);
            cb_wait_front(scaler_cb, 1);

            // Required binary-op LLK init (pi05-llk-binary-op-init-common).
            binary_op_init_common(in_cb, scaler_cb, max_cb);

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
                    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW, /*enforce_fp32=*/true>(
                        exp_cb, scaler_cb, k, 0, 0);
                }
                reduce_uninit();
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, sum_cb);
                tile_regs_release();
                cb_push_back(sum_cb, 1);

                // ============================================================
                // PHASE 4: isum = 1 / sum → isum_cb.
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

                // ============================================================
                // PHASE 5: out = exp * isum (broadcast COL).
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
            }
#endif
        }
    };
};

}  // namespace pi05_siglip_ops
