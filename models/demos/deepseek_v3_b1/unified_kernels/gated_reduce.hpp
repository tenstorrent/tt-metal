// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
// GatedReduce micro-op: SiLU(sum(group1)) * sum(group2)
//
// Performs gated local reduction over two groups of input tiles:
//   1. Reduces group1 tiles with pairwise add, applies SiLU
//   2. Reduces group2 tiles with pairwise add (no activation)
//   3. Multiplies the two results
//
// Produces k_num_tiles output tiles (one per K iteration).
// Each iteration consumes tiles_per_k tiles from each group CB.
//
// CB Layout:
//   group1_cb:   Gate partials (tiles_per_k tiles consumed per iteration)
//   group2_cb:   Up partials (tiles_per_k tiles consumed per iteration)
//   intermed_cb: Intermediate buffer (2 tiles, reused each iteration)
//   out_cb:      Output (1 tile produced per iteration)
// ============================================================================
struct GatedReduce {
    struct ReaderCTArgs {};
    struct WriterCTArgs {};

    template <uint32_t TilesPerK, uint32_t KNumTiles>
    struct ComputeCTArgs {
        static constexpr uint32_t tiles_per_k = TilesPerK;
        static constexpr uint32_t k_num_tiles = KNumTiles;
    };

    struct ReaderArgs {};
    struct WriterArgs {};

    struct ComputeArgs {
        uint32_t group1_cb;    // gate partials CB
        uint32_t group2_cb;    // up partials CB
        uint32_t intermed_cb;  // intermediate CB (2 tiles, reused)
        uint32_t out_cb;       // output CB (1 tile per iteration)
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
            static_assert(tiles_per_k >= 2 && tiles_per_k % 2 == 0, "tiles_per_k must be even and >= 2");

            // Init once before the loop
            binary_op_init_common(args.group1_cb, args.group1_cb, args.intermed_cb);
            silu_tile_init();

            for (uint32_t k = 0; k < k_num_tiles; k++) {
                // Group 1: reduce + SiLU
                add_tiles_init(args.group1_cb, args.group1_cb, true /* acc_to_dest */);

                cb_wait_front(args.group1_cb, tiles_per_k);
                cb_reserve_back(args.intermed_cb, 1);

                tile_regs_acquire();
                for (uint32_t i = 0; i < tiles_per_k; i += 2) {
                    add_tiles(args.group1_cb, args.group1_cb, i, i + 1, 0);
                }
                silu_tile(0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, args.intermed_cb);
                tile_regs_release();

                cb_pop_front(args.group1_cb, tiles_per_k);
                cb_push_back(args.intermed_cb, 1);

                // Group 2: reduce (re-init add for different CB)
                add_tiles_init(args.group2_cb, args.group2_cb, true /* acc_to_dest */);

                cb_wait_front(args.group2_cb, tiles_per_k);
                cb_reserve_back(args.intermed_cb, 1);

                tile_regs_acquire();
                for (uint32_t i = 0; i < tiles_per_k; i += 2) {
                    add_tiles(args.group2_cb, args.group2_cb, i, i + 1, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, args.intermed_cb);
                tile_regs_release();

                cb_pop_front(args.group2_cb, tiles_per_k);
                cb_push_back(args.intermed_cb, 1);

                // Multiply: SiLU(g1) * g2
                cb_wait_front(args.intermed_cb, 2);
                cb_reserve_back(args.out_cb, 1);

                mul_tiles_init(args.intermed_cb, args.intermed_cb);

                tile_regs_acquire();
                mul_tiles(args.intermed_cb, args.intermed_cb, 0, 1, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, args.out_cb);
                tile_regs_release();

                cb_pop_front(args.intermed_cb, 2);
                cb_push_back(args.out_cb, 1);
            }
#endif
        }
    };
};

}  // namespace deepseek_b1_ops
