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
#include "compute_kernel_api/matmul.h"
#include "../kernel_includes/tt_metal/include/compute_kernel_api/custom_mm.h"
#include "compute_kernel_api/tile_move_copy.h"
#endif

namespace deepseek_b1_ops {

// ============================================================================
// Matmul micro-op with configurable output width (supports large out_w via blocking)
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
    // Compile-time args structs - different layout per RISC
    // ========================================================================

    // Reader CTArgs (NCRISC): none
    struct ReaderCTArgs {};

    // Writer CTArgs (BRISC): none
    struct WriterCTArgs {};

    // Compute CTArgs (TRISC): out_w (output width in tiles)
    template <uint32_t out_w_>
    struct ComputeCTArgs {
        static constexpr uint32_t out_w = out_w_;
    };

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
        uint32_t k_num_tiles;
    };

    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

    // ========================================================================
    // Op - the actual operation, templated on CTArgs and IsActiveCore
    // Template args:
    //   CTArgs - compile-time args struct (ReaderCTArgs, WriterCTArgs, or ComputeCTArgs<out_w>)
    //   IsActiveCore - whether this core runs the matmul
    //   pop_in0 - whether to pop in0 after compute (default true)
    //   pop_in1 - whether to pop in1 after compute (default true)
    // ========================================================================
    template <typename CTArgs, bool IsActiveCore, bool pop_in0, bool pop_in1>
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
            constexpr bool transpose = false;
            constexpr uint32_t out_w = CTArgs::out_w;

            // Wait for all input tiles (both from sharded tensors in L1)
            // in1 has num_tiles * out_w tiles (K tiles for each output column)
            cb_wait_front(args.in0, args.k_num_tiles);
            cb_wait_front(args.in1, args.k_num_tiles * out_w);
            // Reserve output tiles
            cb_reserve_back(args.out, out_w);

            if constexpr (out_w <= 8) {
                // Use optimized custom_mm API for up to 8 output tiles (half-DST)
                custom_mm_block_init(args.in0, args.in1, args.out, transpose, args.k_num_tiles);

                tile_regs_acquire();

                for (uint32_t w = 0; w < out_w; w++) {
                    custom_mm_block(args.in0, args.in1, 0, w, w, transpose, args.k_num_tiles, out_w);
                }

                tile_regs_commit();

                tile_regs_wait();
                for (uint32_t w = 0; w < out_w; w++) {
                    pack_tile(w, args.out, w);
                }
                tile_regs_release();
            } else {
                // Use optimized custom_mm API with blocking for out_w > 8
                // Process in blocks of 8 tiles (half-DST with SyncHalf mode)
                constexpr uint32_t dst_tiles = 8;  // Half-DST for reliable blocking
                constexpr uint32_t num_blocks = (out_w + dst_tiles - 1) / dst_tiles;

                for (uint32_t block = 0; block < num_blocks; block++) {
                    uint32_t block_start = block * dst_tiles;
                    uint32_t tiles_in_block = ((block_start + dst_tiles) < out_w) ? dst_tiles : (out_w - block_start);

                    // Re-init for each block to reset MOP state
                    custom_mm_block_init(args.in0, args.in1, args.out, transpose, args.k_num_tiles);

                    tile_regs_acquire();

                    for (uint32_t w = 0; w < tiles_in_block; w++) {
                        custom_mm_block(args.in0, args.in1, 0, block_start + w, w, transpose, args.k_num_tiles, out_w);
                    }

                    tile_regs_commit();

                    tile_regs_wait();
                    for (uint32_t w = 0; w < tiles_in_block; w++) {
                        pack_tile(w, args.out, block_start + w);
                    }
                    tile_regs_release();
                }
            }

            // Pop inputs
            if constexpr (pop_in0) {
                cb_pop_front(args.in0, args.k_num_tiles);
            }
            if constexpr (pop_in1) {
                cb_pop_front(args.in1, args.k_num_tiles * out_w);
            }

            cb_push_back(args.out, out_w);
#endif
        }
    };  // class Op

};  // struct Matmul

}  // namespace deepseek_b1_ops
