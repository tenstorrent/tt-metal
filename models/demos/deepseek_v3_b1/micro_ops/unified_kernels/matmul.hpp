// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC)
#include "dataflow_api.h"
#elif defined(COMPILE_FOR_NCRISC)
#include "dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#include "compute_kernel_api.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"
#endif

namespace deepseek_b1_ops {

// ============================================================================
// Single-tile output Matmul micro-op
//
// Computes: output[1,1] = in0[1,K] @ in1[K,1]
// ============================================================================
struct Matmul {
    // ========================================================================
    // Runtime args structs - different layout per RISC
    // ========================================================================

    // Reader args (NCRISC): [in1, num_tiles]
    struct ReaderArgs {
        uint32_t in1;
        uint32_t num_tiles;
    };

    // Writer args (BRISC): [out]
    struct WriterArgs {
        uint32_t out;
    };

    // Compute args (TRISC): [in0, in1, out, num_tiles]
    struct ComputeArgs {
        uint32_t in0;
        uint32_t in1;
        uint32_t out;
        uint32_t num_tiles;
    };

    using RTArgs = SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

    // ========================================================================
    // Op - the actual operation, templated on IsActiveCore
    // ========================================================================
    template <bool IsActiveCore>
    class Op {
    public:
        void operator()(const RTArgs& args) {
            if constexpr (IsActiveCore) {
                impl(args);
            }
        }

    private:
        void impl(const RTArgs& args) {
#if defined(COMPILE_FOR_NCRISC)
            // ================================================================
            // NCRISC (Reader) - ReaderConfigDescriptor compiles as NCRISC
            // ================================================================
            // Push weights (in1) - backed by sharded tensor
            cb_reserve_back(args.in1, args.num_tiles);
            cb_push_back(args.in1, args.num_tiles);
#elif defined(COMPILE_FOR_BRISC)
            // ================================================================
            // BRISC (Writer) - WriterConfigDescriptor compiles as BRISC
            // ================================================================
            // Wait for output tile to be ready
            // Note: out is backed by sharded tensor, data written directly to L1
            cb_wait_front(args.out, 1);
#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC (Compute)
            // ================================================================
            constexpr uint32_t out_subblock_h = 1;
            constexpr uint32_t out_subblock_w = 1;
            constexpr uint32_t in0_block_w = 1;  // Process one K tile at a time

            // Initialize matmul
            mm_block_init(args.in0, args.in1, args.out, false, out_subblock_w, out_subblock_h, in0_block_w);

            // Wait for all input tiles (both from sharded tensors in L1)
            cb_wait_front(args.in0, args.num_tiles);
            cb_wait_front(args.in1, args.num_tiles);

            // Reserve output
            cb_reserve_back(args.out, 1);

            // Accumulate across K dimension
            tile_regs_acquire();

            for (uint32_t k = 0; k < args.num_tiles; k++) {
                matmul_tiles(args.in0, args.in1, k, k, 0);
            }

            tile_regs_commit();

            // Pop inputs
            cb_pop_front(args.in0, args.num_tiles);
            cb_pop_front(args.in1, args.num_tiles);

            // Pack output
            tile_regs_wait();
            pack_tile(0, args.out);
            tile_regs_release();

            cb_push_back(args.out, 1);
#endif
        }
    };  // class Op

};  // struct Matmul

}  // namespace deepseek_b1_ops
