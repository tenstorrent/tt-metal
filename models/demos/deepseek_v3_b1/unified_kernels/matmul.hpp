// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#include "compute_kernel_api.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"
#endif

namespace deepseek_b1_ops {

// ============================================================================
// Matmul micro-op with configurable output width (up to 4)
//
// Computes: output[1,out_w] = in0[1,K] @ in1[K,out_w]
//
// CB States:
//   NCRISC: No-op (in1 setup done externally via setup_sharded_buffer)
//   BRISC: No-op (next op waits on output if needed)
//   TRISC (Compute):
//     - Waits: in0 (num_tiles), in1 (num_tiles * out_w)
//     - Reserves: out (out_w tiles)
//     - Pushes: out (out_w tiles)
//     - Pops: in0 (num_tiles) if pop_in0=true, in1 (num_tiles * out_w) if pop_in1=true
// ============================================================================
struct Matmul {
    // ========================================================================
    // Runtime args structs - different layout per RISC
    // ========================================================================

    // Reader args (NCRISC): none (NCRISC is no-op, setup done externally)
    struct ReaderArgs {};

    // Writer args (BRISC): none (BRISC is no-op)
    struct WriterArgs {};

    // Compute args (TRISC): [in0, in1, out, num_tiles, out_w]
    struct ComputeArgs {
        uint32_t in0;
        uint32_t in1;
        uint32_t out;
        uint32_t k_num_tiles;
        uint32_t out_w;  // Output width in tiles (up to 4)
    };

    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

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
            // in1 has num_tiles * out_w tiles (K tiles for each output column)
            cb_wait_front(args.in0, args.k_num_tiles);
            cb_wait_front(args.in1, args.k_num_tiles * args.out_w);

            // Reserve output tiles
            cb_reserve_back(args.out, args.out_w);

            // Accumulate all output tiles using different DST indices
            tile_regs_acquire();

            for (uint32_t k = 0; k < args.k_num_tiles; k++) {
                for (uint32_t w = 0; w < args.out_w; w++) {
                    // Each output tile w accumulates into DST[w]
                    matmul_tiles(args.in0, args.in1, k, k * args.out_w + w, w);
                }
            }

            tile_regs_commit();

            // Pack all output tiles
            tile_regs_wait();
            for (uint32_t w = 0; w < args.out_w; w++) {
                pack_tile(w, args.out, w);
            }
            tile_regs_release();

            // Pop inputs
            if constexpr (pop_in0) {
                cb_pop_front(args.in0, args.k_num_tiles);
            }
            if constexpr (pop_in1) {
                cb_pop_front(args.in1, args.k_num_tiles * args.out_w);
            }

            cb_push_back(args.out, args.out_w);
#endif
        }
    };  // class Op

};  // struct Matmul

}  // namespace deepseek_b1_ops
