// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#include <cstdint>
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary.h"
#include "../../../kernel_includes/tt_metal/include/compute_kernel_api/deepseek_moe_gate.h"
#endif

namespace deepseek_b1_ops {

// ============================================================================
// DeepseekMoeGate micro-op
//
// Computes top-8 expert selection with normalized scores.
// Input: router logits [16, 16] + bias [16, 16] + indices [16, 16]
// Output: top8 scores [1, 16] + top8 indices [1, 16] (only first 8 valid)
//
// CB States:
//   NCRISC: Signals tensor-backed input CBs ready (input, bias, indices)
//   BRISC: Waits for output CBs (scores, indices)
//   TRISC: Computes sigmoid (optional), bias add, sorting, normalization
// ============================================================================
struct DeepseekMoeGate {
    // ========================================================================
    // Compile-time args structs
    // ========================================================================

    // Reader CTArgs (NCRISC)
    // skip_input_signal: set to true when input CB is already pushed by another op (e.g., gather)
    template <uint32_t input_cb_, uint32_t bias_cb_, uint32_t input_indices_cb_, bool skip_input_signal_ = false>
    struct ReaderCTArgs {
        static constexpr uint32_t input_cb = input_cb_;
        static constexpr uint32_t bias_cb = bias_cb_;
        static constexpr uint32_t input_indices_cb = input_indices_cb_;
        static constexpr bool skip_input_signal = skip_input_signal_;
    };

    // Writer CTArgs (BRISC)
    template <uint32_t output_cb_, uint32_t output_indices_cb_>
    struct WriterCTArgs {
        static constexpr uint32_t output_cb = output_cb_;
        static constexpr uint32_t output_indices_cb = output_indices_cb_;
    };

    // Compute CTArgs (TRISC)
    // enable_sigmoid must be compile-time (template parameter for deepseek_moe_gate<>)
    template <
        uint32_t input_cb_,
        uint32_t bias_cb_,
        uint32_t input_indices_cb_,
        uint32_t output_cb_,
        uint32_t output_indices_cb_,
        uint32_t eps_,
        uint32_t scaling_factor_,
        uint32_t enable_sigmoid_>
    struct ComputeCTArgs {
        static constexpr uint32_t input_cb = input_cb_;
        static constexpr uint32_t bias_cb = bias_cb_;
        static constexpr uint32_t input_indices_cb = input_indices_cb_;
        static constexpr uint32_t output_cb = output_cb_;
        static constexpr uint32_t output_indices_cb = output_indices_cb_;
        static constexpr uint32_t eps = eps_;
        static constexpr uint32_t scaling_factor = scaling_factor_;
        static constexpr bool enable_sigmoid = enable_sigmoid_ == 1;
    };

    // ========================================================================
    // Op - the actual operation, templated on CTArgs and IsActiveCore
    // ========================================================================
    template <typename CTArgs, bool IsActiveCore>
    class Op {
    public:
        void operator()() {
            if constexpr (IsActiveCore) {
                impl();
            }
        }

    private:
        void impl() {
#if defined(COMPILE_FOR_NCRISC)
            // ================================================================
            // NCRISC: Signal that tensor-backed input CBs are ready
            // ================================================================
            cb_reserve_back(CTArgs::bias_cb, 1);
            cb_push_back(CTArgs::bias_cb, 1);
            cb_reserve_back(CTArgs::input_indices_cb, 1);
            cb_push_back(CTArgs::input_indices_cb, 1);
            // Skip input CB signaling if already pushed by another op (e.g., gather)
            if constexpr (!CTArgs::skip_input_signal) {
                cb_reserve_back(CTArgs::input_cb, 1);
                cb_push_back(CTArgs::input_cb, 1);
            }

#elif defined(COMPILE_FOR_BRISC)
            // ================================================================
            // BRISC: Wait for compute to finish
            // ================================================================
            cb_wait_front(CTArgs::output_indices_cb, 1);
            cb_wait_front(CTArgs::output_cb, 1);

#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC: Compute gate logic
            // ================================================================
            // Init portion
            binary_op_init_common(CTArgs::input_cb, CTArgs::bias_cb, CTArgs::output_cb);
            cb_wait_front(CTArgs::input_indices_cb, 1);
            cb_wait_front(CTArgs::bias_cb, 1);

            // Compute portion
            copy_tile_to_dst_init_short(CTArgs::input_indices_cb);
            reconfig_data_format_srca(CTArgs::input_indices_cb);

            tile_regs_acquire();

            // Copy indices (already transposed to cols)
            copy_tile(CTArgs::input_indices_cb, 0, 1);

            reconfig_data_format_srca(CTArgs::input_cb);
            deepseek_moe_gate_init<CTArgs::enable_sigmoid>(CTArgs::input_cb, CTArgs::bias_cb);
            cb_wait_front(CTArgs::input_cb, 1);
            deepseek_moe_gate<CTArgs::enable_sigmoid>(
                CTArgs::input_cb, CTArgs::bias_cb, CTArgs::eps, CTArgs::scaling_factor);
            // Pop input tile
            cb_pop_front(CTArgs::input_cb, 1);

            tile_regs_commit();

            pack_reconfig_data_format(CTArgs::output_cb);
            cb_reserve_back(CTArgs::output_cb, 1);
            cb_reserve_back(CTArgs::output_indices_cb, 1);

            tile_regs_wait();

            pack_tile(0, CTArgs::output_cb);
            cb_push_back(CTArgs::output_cb, 1);

            pack_reconfig_data_format(CTArgs::output_indices_cb);
            pack_tile(1, CTArgs::output_indices_cb);
            cb_push_back(CTArgs::output_indices_cb, 1);

            tile_regs_release();
#endif
        }
    };  // class Op

};  // struct DeepseekMoeGate

}  // namespace deepseek_b1_ops
