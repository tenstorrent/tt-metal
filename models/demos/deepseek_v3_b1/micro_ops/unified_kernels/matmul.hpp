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
//
// CB States:
//   NCRISC: No-op (in1 setup done externally via setup_sharded_buffer)
//   BRISC: No-op (next op waits on output if needed)
//   TRISC (Compute):
//     - Waits: in0 (num_tiles), in1 (num_tiles)
//     - Reserves: out (1 tile)
//     - Pushes: out (1 tile)
//     - Pops: in0 (num_tiles) if pop_in0=true, in1 (num_tiles) if pop_in1=true
// ============================================================================
struct Matmul {
    // ========================================================================
    // Runtime args structs - different layout per RISC
    // ========================================================================

    // Reader args (NCRISC): none (NCRISC is no-op, setup done externally)
    struct ReaderArgs {};

    // Writer args (BRISC): none (BRISC is no-op)
    struct WriterArgs {};

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
    // Template args:
    //   IsActiveCore - whether this core runs the matmul
    //   pop_in0 - whether to pop in0 after compute (default true)
    //   pop_in1 - whether to pop in1 after compute (default true)
    // ========================================================================
    template <bool IsActiveCore, bool pop_in0, bool pop_in1>
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
            if constexpr (pop_in0) {
                cb_pop_front(args.in0, args.num_tiles);
            }
            if constexpr (pop_in1) {
                cb_pop_front(args.in1, args.num_tiles);
            }

            // Pack output
            tile_regs_wait();
            pack_tile(0, args.out);
            tile_regs_release();

            PACK((DPRINT << TileSlice(args.out, 0, SliceRange::hw0_32_8(), true, true) << ENDL()));
            cb_push_back(args.out, 1);
#endif
        }
    };  // class Op

};  // struct Matmul

}  // namespace deepseek_b1_ops
