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
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reg_api.h"
#endif

namespace deepseek_b1_ops {

// ============================================================================
// RoPE (Rotary Position Embedding) micro-op
//
// Computes: output = (input * cos) + (rotate_half(input) * sin)
// where rotate_half(input) = input @ trans_mat
//
// CB States:
//   NCRISC: Signals sharded input CBs are ready (trans_mat, sin, cos)
//   BRISC: No-op
//   TRISC (Compute): Performs the RoPE computation
// ============================================================================
struct Rope {
    // ========================================================================
    // Compile-time args structs - different layout per RISC
    // ========================================================================

    // Reader CTArgs (NCRISC): Wt and Ht for sharded input signaling
    template <uint32_t Wt_, uint32_t Ht_>
    struct ReaderCTArgs {
        static constexpr uint32_t Wt = Wt_;  // head_dim in tiles
        static constexpr uint32_t Ht = Ht_;  // num_heads per core
    };

    // Writer CTArgs (BRISC): none
    struct WriterCTArgs {};

    // Compute CTArgs (TRISC): Wt and Ht as template parameters
    template <uint32_t Wt_, uint32_t Ht_>
    struct ComputeCTArgs {
        static constexpr uint32_t Wt = Wt_;  // head_dim in tiles
        static constexpr uint32_t Ht = Ht_;  // num_heads per core
    };

    // ========================================================================
    // Runtime args structs - different layout per RISC
    // ========================================================================

    // Reader args (NCRISC): CB indices for sharded input signaling
    struct ReaderArgs {
        uint32_t in_cb;
        uint32_t cos_cb;
        uint32_t sin_cb;
        uint32_t trans_mat_cb;
    };

    // Writer args (BRISC): none
    struct WriterArgs {};

    // Compute args (TRISC): CB indices as runtime args
    struct ComputeArgs {
        uint32_t in_cb;
        uint32_t cos_cb;
        uint32_t sin_cb;
        uint32_t trans_mat_cb;
        uint32_t rotated_in_interm_cb;
        uint32_t cos_interm_cb;
        uint32_t sin_interm_cb;
        uint32_t out_cb;
    };

    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

    // ========================================================================
    // Op - the actual operation, templated on CTArgs, IsActiveCore, and SkipFullInit
    // SkipFullInit: When true, skip full mm_init/binary_op_init_common
    // ========================================================================
    template <typename CTArgs, bool IsActiveCore, bool SkipFullInit = false>
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
            constexpr uint32_t Wt = CTArgs::Wt;
            constexpr uint32_t Ht = CTArgs::Ht;

            // ================================================================
            // Wait for sharded CBs (signaled by NCRISC)
            // ================================================================
            cb_wait_front(args.trans_mat_cb, 1);  // Trans_mat: 1 tile, reused for all heads
            cb_wait_front(args.sin_cb, Wt);       // Sin: Wt tiles (reused for all heads)
            cb_wait_front(args.cos_cb, Wt);       // Cos: Wt tiles (reused for all heads)
            // ================================================================
            // Initialize matmul and binary ops
            // In fused kernels (SkipFullInit=true), skip full init because multiple full
            // inits are unsafe (can interfere with other matmul operations on the same core).
            // Use mm_init_short in the loop to reconfigure CB pointers as needed.
            // ================================================================
            if constexpr (!SkipFullInit) {
                mm_init(args.in_cb, args.trans_mat_cb, args.rotated_in_interm_cb);
                binary_op_init_common(args.rotated_in_interm_cb, args.sin_cb, args.sin_interm_cb);
            }
            // ================================================================
            // Main loop: process Ht heads, each head consumes Wt tiles
            // ================================================================
            for (uint32_t ht = 0; ht < Ht; ht++) {
                // Reserve intermediate and output buffers
                cb_reserve_back(args.rotated_in_interm_cb, Wt);
                cb_reserve_back(args.sin_interm_cb, Wt);
                cb_reserve_back(args.cos_interm_cb, Wt);
                cb_reserve_back(args.out_cb, Wt);

                cb_wait_front(args.in_cb, Wt);

                // ============================================================
                // Step 1: rotated = input @ trans_mat (matmul for rotate_half)
                // ============================================================
                mm_init_short(args.in_cb, args.trans_mat_cb);
                tile_regs_acquire();
                for (uint32_t j = 0; j < Wt; ++j) {
                    matmul_tiles(args.in_cb, args.trans_mat_cb, j, 0, j);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < Wt; ++j) {
                    pack_tile(j, args.rotated_in_interm_cb, j);
                }
                tile_regs_release();
                cb_push_back(args.rotated_in_interm_cb, Wt);
                cb_wait_front(args.rotated_in_interm_cb, Wt);

                // ============================================================
                // Step 2: sin_interm = rotated * sin (broadcast multiply)
                // ============================================================
                mul_bcast_rows_init_short(args.rotated_in_interm_cb, args.sin_cb);
                tile_regs_acquire();
                for (uint32_t j = 0; j < Wt; ++j) {
                    mul_tiles_bcast<BroadcastType::ROW>(args.rotated_in_interm_cb, args.sin_cb, j, j, j);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < Wt; ++j) {
                    pack_tile(j, args.sin_interm_cb, j);
                }
                tile_regs_release();
                cb_push_back(args.sin_interm_cb, Wt);
                cb_pop_front(args.rotated_in_interm_cb, Wt);

                // ============================================================
                // Step 3: cos_interm = input * cos (broadcast multiply)
                // ============================================================
                tile_regs_acquire();
                for (uint32_t j = 0; j < Wt; ++j) {
                    mul_tiles_bcast<BroadcastType::ROW>(args.in_cb, args.cos_cb, j, j, j);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < Wt; ++j) {
                    pack_tile(j, args.cos_interm_cb, j);
                }
                tile_regs_release();
                cb_push_back(args.cos_interm_cb, Wt);
                cb_pop_front(args.in_cb, Wt);

                // ============================================================
                // Step 4: output = cos_interm + sin_interm (add)
                // ============================================================
                cb_wait_front(args.sin_interm_cb, Wt);
                cb_wait_front(args.cos_interm_cb, Wt);
                add_tiles_init(args.cos_interm_cb, args.sin_interm_cb);
                tile_regs_acquire();
                for (uint32_t j = 0; j < Wt; ++j) {
                    add_tiles(args.cos_interm_cb, args.sin_interm_cb, j, j, j);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < Wt; ++j) {
                    pack_tile(j, args.out_cb, j);
                }
                tile_regs_release();
                cb_push_back(args.out_cb, Wt);
                cb_pop_front(args.sin_interm_cb, Wt);
                cb_pop_front(args.cos_interm_cb, Wt);
            }

            // ================================================================
            // Cleanup: pop sin/cos (trans_mat is reused, not popped)
            // Note: sin/cos are reused for all heads, so only pop once after all heads processed
            // ================================================================
            cb_pop_front(args.sin_cb, Wt);
            cb_pop_front(args.cos_cb, Wt);
#endif
        }
    };
};

}  // namespace deepseek_b1_ops
