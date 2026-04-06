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
// LocalReduce micro-op: Element-wise sum reduction of N tiles with optional SiLU
//
// Computes: output = in_cb[0] + in_cb[1] + ... + in_cb[n-1]
//
// With optional SiLU: output = SiLU(result)
//
// All input tiles come from the same circular buffer.
// Uses acc_to_dest mode for efficient pairwise accumulation.
//
// CB Layout:
//   CB0 (in):  Contains N tiles to reduce
//   CB1 (out): Output (1 tile)
// ============================================================================
struct LocalReduce {
    struct ReaderCTArgs {};
    struct WriterCTArgs {};
    template <uint32_t NumTiles, bool ApplySilu>
    struct ComputeCTArgs {
        static constexpr uint32_t num_tiles = NumTiles;
        static constexpr bool apply_silu = ApplySilu;
    };

    struct ReaderArgs {};
    struct WriterArgs {};

    struct ComputeArgs {
        uint32_t in_cb;   // Input CB with N tiles
        uint32_t out_cb;  // Output CB
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
            constexpr uint32_t num_tiles = CTArgs::num_tiles;
            constexpr bool apply_silu = CTArgs::apply_silu;

            // Initialize operations before waiting for data
            reconfig_data_format<false, true>(args.in_cb, args.in_cb);
            pack_reconfig_data_format<true>(args.out_cb);
            add_tiles_init(args.in_cb, args.in_cb, true /* acc_to_dest */);
            if constexpr (apply_silu) {
                silu_tile_init();
            }

            // Wait for all input tiles
            cb_wait_front(args.in_cb, num_tiles);

            // Reserve output
            cb_reserve_back(args.out_cb, 1);

            // Acquire dest register
            tile_regs_acquire();

            // Sum all tiles using acc_to_dest mode
            // acc_to_dest=true means: DST[0] = A + B + DST[0]
            // DST accumulator starts at zero for each tile position
            for (uint32_t i = 0; i < num_tiles; i += 2) {
                add_tiles(args.in_cb, args.in_cb, i, i + 1, 0);
            }

            // Optionally apply SiLU activation
            if constexpr (apply_silu) {
                silu_tile(0);
            }

            // Commit and wait for compute
            tile_regs_commit();
            tile_regs_wait();

            // Pack result
            pack_tile(0, args.out_cb);

            // Release dest register
            tile_regs_release();

            // Pop inputs and push output
            cb_pop_front(args.in_cb, num_tiles);
            cb_push_back(args.out_cb, 1);
#endif
        }
    };
};

}  // namespace deepseek_b1_ops
