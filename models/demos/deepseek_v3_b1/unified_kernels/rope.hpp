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
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/reg_api.h"
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

    // Reader CTArgs (NCRISC): Wt for sharded input signaling
    template <uint32_t Wt_>
    struct ReaderCTArgs {
        static constexpr uint32_t Wt = Wt_;  // head_dim in tiles
    };

    // Writer CTArgs (BRISC): none
    struct WriterCTArgs {};

    // Compute CTArgs (TRISC): Wt as template parameter (Ht=1 hardcoded)
    template <uint32_t Wt_>
    struct ComputeCTArgs {
        static constexpr uint32_t Wt = Wt_;  // head_dim in tiles
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
    // Op - the actual operation, templated on CTArgs and IsActiveCore
    // ========================================================================
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
#if defined(COMPILE_FOR_NCRISC)
            constexpr uint32_t Ht = 1;
            constexpr uint32_t Wt = CTArgs::Wt;

            for (uint32_t ht = 0; ht < Ht; ht++) {
                cb_reserve_back(args.in_cb, Wt);
                cb_push_back(args.in_cb, Wt);
            }

            cb_reserve_back(args.trans_mat_cb, 1);
            cb_push_back(args.trans_mat_cb, 1);

            cb_reserve_back(args.sin_cb, Wt);
            cb_push_back(args.sin_cb, Wt);

            cb_reserve_back(args.cos_cb, Wt);
            cb_push_back(args.cos_cb, Wt);
#elif defined(COMPILE_FOR_TRISC)
            constexpr uint32_t Wt = CTArgs::Wt;
            constexpr uint32_t Ht = 1;

            // ================================================================
            // Wait for sharded CBs (signaled by NCRISC)
            // ================================================================
            cb_wait_front(args.trans_mat_cb, 1);  // Trans_mat: 1 tile, reused for all heads
            cb_wait_front(args.sin_cb, Wt);       // Sin: Wt tiles
            cb_wait_front(args.cos_cb, Wt);       // Cos: Wt tiles

            // ================================================================
            // Initialize matmul and binary ops (done once before loop)
            // ================================================================
            mm_init(args.in_cb, args.trans_mat_cb, args.rotated_in_interm_cb);
            binary_op_init_common(args.rotated_in_interm_cb, args.sin_cb, args.sin_interm_cb);

            // ================================================================
            // Main loop: process each head tile row
            // ================================================================
            for (uint32_t ht = 0; ht < Ht; ht++) {
                // Reserve intermediate and output buffers
                cb_reserve_back(args.rotated_in_interm_cb, Wt);
                cb_reserve_back(args.sin_interm_cb, Wt);
                cb_reserve_back(args.cos_interm_cb, Wt);
                cb_reserve_back(args.out_cb, Wt);

                // Signal input row is ready (sharded tensor)
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
            // ================================================================
            cb_pop_front(args.sin_cb, Wt);
            cb_pop_front(args.cos_cb, Wt);
#endif
        }
    };
};

}  // namespace deepseek_b1_ops
