// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// SigLIP encoder-shape matmul Op-struct.
//
// Ported from siglip qkv_matmul_kernel.cpp (the shape-agnostic encoder kernel
// reused by QKV, O-proj, FC1, and FC2 path-2 matmul cases). Computes per core:
//   output[M_TILES, N_TILES_PER_CORE] =
//       activation[M_TILES, K_TILES] @ weights[K_TILES, N_TILES_PER_CORE]
//
// SigLIP layer-0 shapes:
//   QKV:    M=8, K=36, N_per_core=3  (36 cores × 3 = 108 N-tiles = N=3456)
//   O-proj: M=8, K=36, N_per_core=1  (36 cores × 1 = 36  N-tiles = N=1152)
//
// SUBBLOCK_H = 1 is mandatory: SUBBLOCK_H > 1 gave PCC = 0.757 (root cause
// in in0_index stride semantics — open perf opportunity, but PCC is correct
// at H=1). Activation: bf16, HEIGHT_SHARDED, full M×K replicated per core.
// Weight: bfp8, WIDTH_SHARDED, K rows × (N/num_cores) cols per core.
// Output: bf16, WIDTH_SHARDED, same N partition as weight.

#pragma once

#include "../../../../demos/deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#endif

namespace pi05_siglip_ops {

struct EncoderMatmul {
    struct ReaderCTArgs {};
    struct WriterCTArgs {};

    template <
        uint32_t ActCb,
        uint32_t WeightsCb,
        uint32_t OutCb,
        uint32_t MTiles,
        uint32_t KTiles,
        uint32_t NTilesPerCore,
        uint32_t ActTotalTiles,
        uint32_t WeightTiles>
    struct ComputeCTArgs {
        static constexpr uint32_t act_cb = ActCb;
        static constexpr uint32_t weights_cb = WeightsCb;
        static constexpr uint32_t out_cb = OutCb;
        static constexpr uint32_t m_tiles = MTiles;
        static constexpr uint32_t k_tiles = KTiles;
        static constexpr uint32_t n_tiles_per_core = NTilesPerCore;
        static constexpr uint32_t act_total_tiles = ActTotalTiles;
        static constexpr uint32_t weight_tiles = WeightTiles;
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
            constexpr uint32_t act_cb = CTArgs::act_cb;
            constexpr uint32_t weights_cb = CTArgs::weights_cb;
            constexpr uint32_t out_cb = CTArgs::out_cb;
            constexpr uint32_t M_TILES = CTArgs::m_tiles;
            constexpr uint32_t K_TILES = CTArgs::k_tiles;
            constexpr uint32_t N_TILES_PER_CORE = CTArgs::n_tiles_per_core;
            constexpr uint32_t ACT_TOTAL_TILES = CTArgs::act_total_tiles;
            constexpr uint32_t WEIGHT_TILES = CTArgs::weight_tiles;

            // SUBBLOCK_H=1 is load-bearing: SUBBLOCK_H>1 gave PCC 0.757 (open
            // perf opportunity, root cause TBD in in0_index stride semantics).
            constexpr uint32_t SUBBLOCK_H = 1;
            constexpr uint32_t SUBBLOCK_W = N_TILES_PER_CORE;
            constexpr uint32_t OUT_TILES_PER_SUBBLOCK = SUBBLOCK_H * SUBBLOCK_W;
            static_assert(M_TILES % SUBBLOCK_H == 0, "M_TILES must be a multiple of SUBBLOCK_H");
            static_assert(N_TILES_PER_CORE % SUBBLOCK_W == 0, "N_TILES_PER_CORE must be a multiple of SUBBLOCK_W");
            static_assert(OUT_TILES_PER_SUBBLOCK <= 16, "subblock exceeds dst register capacity");
            constexpr uint32_t OUT_TOTAL_TILES = M_TILES * N_TILES_PER_CORE;

            reconfig_data_format(weights_cb, act_cb);
            pack_reconfig_data_format(out_cb);

            mm_init(act_cb, weights_cb, out_cb);

            cb_wait_front(act_cb, ACT_TOTAL_TILES);
            cb_wait_front(weights_cb, WEIGHT_TILES);
            cb_reserve_back(out_cb, OUT_TOTAL_TILES);

            for (uint32_t m_start = 0; m_start < M_TILES; m_start += SUBBLOCK_H) {
                mm_block_init_short(
                    act_cb,
                    weights_cb,
                    /*transpose=*/0,
                    /*ct_dim=*/SUBBLOCK_W,
                    /*rt_dim=*/SUBBLOCK_H,
                    /*kt_dim=*/K_TILES);

                tile_regs_acquire();

                uint32_t in0_index = m_start * K_TILES;
                uint32_t in1_index = 0;
                for (uint32_t k = 0; k < K_TILES; ++k) {
                    matmul_block(
                        act_cb,
                        weights_cb,
                        in0_index,
                        in1_index,
                        /*idst=*/0,
                        /*transpose=*/false,
                        /*ct_dim=*/SUBBLOCK_W,
                        /*rt_dim=*/SUBBLOCK_H,
                        /*kt_dim=*/K_TILES);
                    in0_index += 1;
                    in1_index += N_TILES_PER_CORE;
                }

                tile_regs_commit();

                tile_regs_wait();
                for (uint32_t h = 0; h < SUBBLOCK_H; ++h) {
                    for (uint32_t w = 0; w < SUBBLOCK_W; ++w) {
                        uint32_t dst_idx = h * SUBBLOCK_W + w;
                        uint32_t out_tile_id = (m_start + h) * N_TILES_PER_CORE + w;
                        pack_tile<true>(dst_idx, out_cb, out_tile_id);
                    }
                }
                tile_regs_release();
            }

            cb_push_back(out_cb, OUT_TOTAL_TILES);
            cb_pop_front(act_cb, ACT_TOTAL_TILES);
            // Weights stay L1-resident — no cb_pop_front for weights_cb.
#endif
        }
    };
};

}  // namespace pi05_siglip_ops
