// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/tile_move_copy.h"
#endif

namespace deepseek_b1_ops {

// ============================================================================
// GatedReduce micro-op: SiLU(sum(group1_e)) * sum(group2_e), per expert.
//
// Performs gated local reduction over two groups of input tiles for each of
// `num_experts` experts (per-expert blocks concatenated in each group CB):
//   1. Per expert: reduce group1 tiles with pairwise add, apply SiLU
//   2. Per expert: reduce group2 tiles with pairwise add (no activation)
//   3. Per expert: multiply and emit one output tile
//
// num_experts == 1 produces SiLU(sum(g1)) * sum(g2) — identical to the
// shared-expert behavior. num_experts > 1 is the routed/SRAM multi-expert path.
//
// Produces num_experts * k_num_tiles output tiles. Each expert/K iteration
// consumes tiles_per_k tiles from each group CB.
//
// CB Layout:
//   group1_cb:   Gate partials (tiles_per_k * num_experts tiles per iteration)
//   group2_cb:   Up partials (tiles_per_k * num_experts tiles per iteration)
//   intermed_cb: Intermediate buffer (2 tiles, reused each iteration)
//   out_cb:      Output (num_experts * k_num_tiles tiles produced)
// ============================================================================
struct GatedReduce {
    struct ReaderCTArgs {};
    struct WriterCTArgs {};

    template <uint32_t TilesPerK, uint32_t KNumTiles, uint32_t NumExperts = 1>
    struct ComputeCTArgs {
        static constexpr uint32_t tiles_per_k = TilesPerK;
        static constexpr uint32_t k_num_tiles = KNumTiles;
        static constexpr uint32_t num_experts = NumExperts;
    };

    struct ReaderArgs {};
    struct WriterArgs {};

    struct ComputeArgs {
        uint32_t group1_cb;    // gate partials CB
        uint32_t group2_cb;    // up partials CB
        uint32_t intermed_cb;  // intermediate CB (2 tiles, reused)
        uint32_t out_cb;       // output CB (num_experts * k_num_tiles tiles)
    };

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
        void impl(const RTArgs& args) {
#if defined(COMPILE_FOR_TRISC)
            constexpr uint32_t tiles_per_k = CTArgs::tiles_per_k;
            constexpr uint32_t k_num_tiles = CTArgs::k_num_tiles;
            constexpr uint32_t num_experts = CTArgs::num_experts;
            static_assert(tiles_per_k >= 2 && tiles_per_k % 2 == 0, "tiles_per_k must be even and >= 2");
            static_assert(num_experts >= 1, "num_experts must be >= 1");
            static_assert(num_experts <= 8, "GatedReduce supports up to 8 experts");
            constexpr uint32_t expert_stride_tiles = k_num_tiles * tiles_per_k;
            constexpr uint32_t total_in_tiles = num_experts * expert_stride_tiles;

            // Init once before the loop
            // Assumes all input cbs are configured the same, and the intermediate cb is configured the same as the
            // output cb
            reconfig_data_format<false, true>(args.group1_cb, args.group1_cb);
            pack_reconfig_data_format<true>(args.out_cb);
            silu_tile_init();

            cb_wait_front(args.group1_cb, total_in_tiles);
            cb_wait_front(args.group2_cb, total_in_tiles);

            for (uint32_t e = 0; e < num_experts; e++) {
                for (uint32_t k = 0; k < k_num_tiles; k++) {
                    const uint32_t in_base = e * expert_stride_tiles + k * tiles_per_k;

                    // Group 1: reduce K-slice partials for this expert, then SiLU.
                    add_tiles_init(args.group1_cb, args.group1_cb, true /* acc_to_dest */);
                    cb_reserve_back(args.intermed_cb, 1);

                    tile_regs_acquire();
                    for (uint32_t i = 0; i < tiles_per_k; i += 2) {
                        add_tiles(args.group1_cb, args.group1_cb, in_base + i, in_base + i + 1, 0);
                    }
                    silu_tile(0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, args.intermed_cb);
                    tile_regs_release();
                    cb_push_back(args.intermed_cb, 1);

                    // Group 2: reduce K-slice partials for the same expert.
                    cb_reserve_back(args.intermed_cb, 1);

                    tile_regs_acquire();
                    for (uint32_t i = 0; i < tiles_per_k; i += 2) {
                        add_tiles(args.group2_cb, args.group2_cb, in_base + i, in_base + i + 1, 0);
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, args.intermed_cb);
                    tile_regs_release();
                    cb_push_back(args.intermed_cb, 1);

                    // Multiply: SiLU(gate_e) * up_e. Keep experts separate for down-proj.
                    mul_tiles_init(args.intermed_cb, args.intermed_cb);
                    cb_wait_front(args.intermed_cb, 2);
                    cb_reserve_back(args.out_cb, 1);

                    tile_regs_acquire();
                    mul_tiles(args.intermed_cb, args.intermed_cb, 0, 1, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, args.out_cb);
                    tile_regs_release();

                    cb_pop_front(args.intermed_cb, 2);
                    cb_push_back(args.out_cb, 1);
                }
            }

            cb_pop_front(args.group1_cb, total_in_tiles);
            cb_pop_front(args.group2_cb, total_in_tiles);
#endif
        }
    };
};

}  // namespace deepseek_b1_ops
