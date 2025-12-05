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
    // Compile-time args structs - different layout per RISC
    // ========================================================================

    // Reader CTArgs: [in0_cb, in1_cb, num_tiles_k]
    template <uint32_t In0CB, uint32_t In1CB, uint32_t NumTilesK>
    struct ReaderCTArgs {
        static constexpr uint32_t in0_cb = In0CB;
        static constexpr uint32_t in1_cb = In1CB;
        static constexpr uint32_t num_tiles_k = NumTilesK;
    };

    // Writer CTArgs: [out_cb]
    template <uint32_t OutCB>
    struct WriterCTArgs {
        static constexpr uint32_t out_cb = OutCB;
    };

    // Compute CTArgs: [in0_cb, in1_cb, out_cb, interm_cb, num_tiles_k, fp32_acc]
    template <uint32_t In0CB, uint32_t In1CB, uint32_t OutCB, uint32_t IntermCB, uint32_t NumTilesK, bool FP32Acc>
    struct ComputeCTArgs {
        static constexpr uint32_t in0_cb = In0CB;
        static constexpr uint32_t in1_cb = In1CB;
        static constexpr uint32_t out_cb = OutCB;
        static constexpr uint32_t interm_cb = IntermCB;
        static constexpr uint32_t num_tiles_k = NumTilesK;
        static constexpr bool fp32_acc = FP32Acc;
    };

    // ========================================================================
    // Op - the actual operation, templated on CTArgs
    // ========================================================================
    template <typename CTArgs>
    class Op {
    public:
        // ====================================================================
        // Phase-specific RTArgs (none for this simple matmul)
        // ====================================================================
        struct ReaderArgs {};
        struct WriterArgs {};
        struct ComputeArgs {};

        using RTArgs = SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

        void operator()(const RTArgs& = {}) { impl(); }

    private:
        void impl() {
#if defined(COMPILE_FOR_NCRISC)
            // ================================================================
            // NCRISC (Reader) - ReaderConfigDescriptor compiles as NCRISC
            // ================================================================
            // Both in0 and in1 are backed by sharded tensors - just signal they're ready
            cb_reserve_back(CTArgs::in0_cb, CTArgs::num_tiles_k);
            cb_push_back(CTArgs::in0_cb, CTArgs::num_tiles_k);

            cb_reserve_back(CTArgs::in1_cb, CTArgs::num_tiles_k);
            cb_push_back(CTArgs::in1_cb, CTArgs::num_tiles_k);
#elif defined(COMPILE_FOR_BRISC)
            // ================================================================
            // BRISC (Writer) - WriterConfigDescriptor compiles as BRISC
            // ================================================================
            // Wait for output tile to be ready
            // Note: out_cb is backed by sharded tensor, data written directly to L1
            cb_wait_front(CTArgs::out_cb, 1);
#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC (Compute)
            // ================================================================
            constexpr uint32_t out_subblock_h = 1;
            constexpr uint32_t out_subblock_w = 1;
            constexpr uint32_t in0_block_w = 1;  // Process one K tile at a time

            // Initialize matmul
            mm_block_init(
                CTArgs::in0_cb, CTArgs::in1_cb, CTArgs::out_cb, false, out_subblock_w, out_subblock_h, in0_block_w);

            // Wait for all input tiles (both from sharded tensors in L1)
            cb_wait_front(CTArgs::in0_cb, CTArgs::num_tiles_k);
            cb_wait_front(CTArgs::in1_cb, CTArgs::num_tiles_k);

            // Reserve output
            cb_reserve_back(CTArgs::out_cb, 1);

            // Accumulate across K dimension
            tile_regs_acquire();

            for (uint32_t k = 0; k < CTArgs::num_tiles_k; k++) {
                matmul_tiles(CTArgs::in0_cb, CTArgs::in1_cb, k, k, 0, false);
            }

            tile_regs_commit();

            // Pop inputs
            cb_pop_front(CTArgs::in0_cb, CTArgs::num_tiles_k);
            cb_pop_front(CTArgs::in1_cb, CTArgs::num_tiles_k);

            // Pack output
            tile_regs_wait();
            pack_tile(0, CTArgs::out_cb);
            tile_regs_release();

            cb_push_back(CTArgs::out_cb, 1);
#endif
        }
    };  // class Op

};  // struct Matmul

}  // namespace deepseek_b1_ops
