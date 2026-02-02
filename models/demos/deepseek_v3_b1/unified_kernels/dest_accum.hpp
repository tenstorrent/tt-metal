// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#endif

namespace deepseek_b1_ops {

// ============================================================================
// DestAccum micro-op: Element-wise addition of N tiles from single CB
//
// Computes: output = in_cb[0] + in_cb[1] + ... + in_cb[n-1]
//
// All input tiles come from the same circular buffer.
// Uses acc_to_dest mode with DST zero-initialization at kernel start.
//
// CB Layout:
//   CB0 (in):  Contains N tiles to accumulate
//   CB1 (out): Output (1 tile)
// ============================================================================
struct DestAccum {
    struct ReaderCTArgs {};
    struct WriterCTArgs {};
    struct ComputeCTArgs {
        uint32_t num_tiles;  // Number of tiles to accumulate
    };

    struct ReaderArgs {};
    struct WriterArgs {};

    struct ComputeArgs {
        uint32_t in_cb;      // Input CB with N tiles
        uint32_t out_cb;     // Output CB
        uint32_t num_tiles;  // Number of tiles to accumulate
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
            const uint32_t num_tiles = args.num_tiles;

            // Wait for all input tiles
            cb_wait_front(args.in_cb, num_tiles);

            // Reserve output
            cb_reserve_back(args.out_cb, 1);

            // Initialize binary add operation
            binary_op_init_common(args.in_cb, args.in_cb, args.out_cb);

            // Acquire dest register (DST is zeroed at kernel startup)
            tile_regs_acquire();

            add_tiles_init(args.in_cb, args.in_cb, true /* acc_to_dest */);
            for (uint32_t i = 0; i < num_tiles; i += 2) {
                add_tiles(args.in_cb, args.in_cb, i, i + 1, 0);
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
