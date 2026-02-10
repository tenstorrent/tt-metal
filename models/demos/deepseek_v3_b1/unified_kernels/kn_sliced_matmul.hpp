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
#include "../kernel_includes/tt_metal/include/compute_kernel_api/custom_mm.h"
#include "api/compute/tile_move_copy.h"
#endif

namespace deepseek_b1_ops {

// ============================================================================
// KNSlicedMatmul micro-op: KN-parallel partial matmul with offset into activation buffer
//
// Computes: output[1,out_w] = act[k_offset..k_offset+k_per_core] @ weights[k_per_core,out_w]
//
// Each core takes a slice of the shared activation buffer (via k_offset) and
// multiplies it against its local weight shard, producing out_w output tiles.
// Fixed properties: transpose=false, split_acc=true, dense_packing=true.
//
// CB States:
//   NCRISC: No-op (weights setup done externally via setup_sharded_buffer)
//   BRISC: No-op
//   TRISC (Compute):
//     - Waits: act_cb (act_total_tiles), weights_cb (k_per_core)
//     - Reserves: out_cb (out_w tiles)
//     - Pushes: out_cb (out_w tiles)
//     - Pops: act_cb (act_total_tiles) if pop_act=true,
//             weights_cb (k_per_core) if pop_weights=true
// ============================================================================
struct KNSlicedMatmul {
    // ========================================================================
    // Compile-time args structs - different layout per RISC
    // ========================================================================

    // Reader CTArgs (NCRISC): none
    struct ReaderCTArgs {};

    // Writer CTArgs (BRISC): none
    struct WriterCTArgs {};

    // Compute CTArgs (TRISC)
    template <uint32_t OutW = 1>
    struct ComputeCTArgs {
        static constexpr uint32_t out_w = OutW;
    };

    // ========================================================================
    // Runtime args structs - different layout per RISC
    // ========================================================================

    // Reader args (NCRISC): none
    struct ReaderArgs {};

    // Writer args (BRISC): none
    struct WriterArgs {};

    // Compute args (TRISC)
    struct ComputeArgs {
        uint32_t act_cb;           // activation CB (full shared buffer)
        uint32_t weights_cb;       // weights CB (per-core K-slice)
        uint32_t out_cb;           // output CB (out_w tiles)
        uint32_t k_offset;         // tile offset into act_cb
        uint32_t k_per_core;       // K tiles this core processes
        uint32_t act_total_tiles;  // total tiles in act_cb (for wait/pop)
    };

    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

    // ========================================================================
    // Op - the actual operation, templated on CTArgs and IsActiveCore
    // Template args:
    //   CTArgs - compile-time args struct (ReaderCTArgs, WriterCTArgs, or ComputeCTArgs)
    //   IsActiveCore - whether this core runs the matmul
    //   pop_act - whether to pop act_cb after compute (default true)
    //   pop_weights - whether to pop weights_cb after compute (default false)
    // ========================================================================
    template <typename CTArgs, bool IsActiveCore, bool pop_act = true, bool pop_weights = false>
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
            constexpr bool transpose = false;
            constexpr bool split_acc = true;
            constexpr bool dense_packing = true;
            constexpr bool finalize = true;
            constexpr uint32_t out_w = CTArgs::out_w;

            custom_mm_block_init<transpose, split_acc, dense_packing>(args.act_cb, args.weights_cb, args.out_cb, out_w);

            // Wait for all activation tiles and weight tiles
            cb_wait_front(args.act_cb, args.act_total_tiles);
            cb_wait_front(args.weights_cb, args.k_per_core);

            // Reserve output tile
            cb_reserve_back(args.out_cb, out_w);

            tile_regs_acquire();
            custom_mm_block<finalize, false>(args.act_cb, args.weights_cb, args.k_offset, 0, 0, args.k_per_core, out_w);
            tile_regs_commit();

            tile_regs_wait();
            for (uint32_t j = 0; j < out_w; j++) {
                pack_tile(j, args.out_cb, j);
            }
            tile_regs_release();

            custom_mm_block_uninit<dense_packing>();

            // Pop inputs
            if constexpr (pop_act) {
                cb_pop_front(args.act_cb, args.act_total_tiles);
            }
            if constexpr (pop_weights) {
                cb_pop_front(args.weights_cb, args.k_per_core);
            }

            cb_push_back(args.out_cb, out_w);
#endif
        }
    };  // class Op

};  // struct KNSlicedMatmul

}  // namespace deepseek_b1_ops
