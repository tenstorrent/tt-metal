// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#include "../kernel_includes/tt_metal/dm_utils.hpp"
#elif defined(COMPILE_FOR_TRISC)
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_SCALAR
#include "api/compute/compute_kernel_api.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/experimental/mul_reduce_scalar.h"
#include "../kernel_includes/tt_metal/include/compute_kernel_api/add_rsqrt.h"
#include "../kernel_includes/tt_metal/include/compute_kernel_api/rmsnorm.h"
#endif

namespace deepseek_b1_ops {

// ============================================================================
// RMSNorm micro-op
//
// Computes: output = (input / RMS(input)) * gamma
// Where RMS(x) = sqrt(mean(x^2) + epsilon)
//
// CB States:
//   NCRISC (Reader):
//     Note: input_cb and gamma_cb setup done externally via setup_sharded_buffer
//   BRISC: No-op (next op waits on output if needed)
//   TRISC (Compute):
//     - Waits: input_cb (num_tiles), gamma_cb (num_tiles)
//     - Reserves: output_cb (num_tiles)
//     - Pushes: output_cb (num_tiles)
//     - Pops: input_cb (num_tiles) if pop_input=true
// ============================================================================
struct RMSNorm {
    // ========================================================================
    // Compile-time args structs - only what MUST be compile-time
    // (used as template parameters or in constexpr expressions)
    // ========================================================================

    // Reader CTArgs:none needed
    struct ReaderCTArgs {};

    // Writer CTArgs: none needed
    struct WriterCTArgs {};

    // Compute CTArgs: fp32_acc, num_tiles, rsqrt_fast_approx (template params)
    template <
        bool FP32Acc,
        uint32_t NumTiles,
        bool RsqrtFastApprox,
        uint32_t InputCb,
        uint32_t GammaCb,
        uint32_t OutputCb>
    struct ComputeCTArgs {
        static constexpr bool fp32_acc = FP32Acc;
        static constexpr uint32_t num_tiles = NumTiles;
        static constexpr bool rsqrt_fast_approx = RsqrtFastApprox;
        static constexpr uint32_t input_cb = InputCb;
        static constexpr uint32_t gamma_cb = GammaCb;
        static constexpr uint32_t output_cb = OutputCb;
    };

    // ========================================================================
    // Runtime args structs - different layout per RISC
    // ========================================================================
    // Reader args (NCRISC): none needed
    struct ReaderArgs {};
    // Writer args (BRISC): none (BRISC is no-op)
    struct WriterArgs {};
    struct ComputeArgs {
        uint32_t epsilon;
        float scalar;
    };

    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

    // ========================================================================
    // Op - the actual operation, templated on CTArgs, IsActiveCore, pop_input
    // ========================================================================
    template <typename CTArgs, bool IsActiveCore, bool pop_input>
    class Op {
    public:
        void operator()(const RTArgs& args) {
            if constexpr (IsActiveCore) {
                impl(args);
            }
        }

    private:
        void impl([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC (Compute)
            // ================================================================
            // Init block done only once
            cb_wait_front(CTArgs::gamma_cb, CTArgs::num_tiles);  // we don't pop, only wait once and reuse

            compute_rmsnorm(args);
#endif
        }

#if defined(COMPILE_FOR_TRISC)
        void compute_rmsnorm(const ComputeArgs& args) {
            constexpr uint32_t num_tiles = CTArgs::num_tiles;
            reconfig_data_format<false, true>(CTArgs::input_cb, CTArgs::input_cb);
            pack_reconfig_data_format<true>(CTArgs::output_cb);
            {
                // Square the input
                mul_reduce_scalar_init(CTArgs::input_cb, CTArgs::input_cb);
                add_rsqrt_tile_init();
                cb_wait_front(CTArgs::input_cb, num_tiles);
                tile_regs_acquire();
                mul_reduce_scalar_tile<PoolType::SUM>(CTArgs::input_cb, CTArgs::input_cb, num_tiles, args.scalar);
                mul_reduce_scalar_uninit();
            }
            {
                add_rsqrt_tile<CTArgs::rsqrt_fast_approx, VectorMode::RC_custom, 1>(0, args.epsilon);
            }
            {
                // Multiply input by 1/RMS
                rmsnorm_mul_bcast_scalar_reuse_tiles_init<num_tiles>(CTArgs::input_cb);
                rmsnorm_mul_bcast_scalar_reuse_tiles<num_tiles, true>(CTArgs::input_cb, 0, 0, 0);
                if constexpr (pop_input) {
                    cb_pop_front(CTArgs::input_cb, num_tiles);
                }
            }
            {
                // Multiply by the weight
                cb_reserve_back(CTArgs::output_cb, num_tiles);
                binary_dest_reuse_tiles_init<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CTArgs::gamma_cb);
                for (uint32_t i = 0; i < num_tiles; i++) {
                    binary_dest_reuse_tiles<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CTArgs::gamma_cb, i, i);
                }

                tile_regs_commit();
                tile_regs_wait();
                pack_tile_block(0, CTArgs::output_cb, num_tiles);
                cb_push_back(CTArgs::output_cb, num_tiles);
                tile_regs_release();
            }
        }
#endif
    };  // class Op

};  // struct RMSNorm

}  // namespace deepseek_b1_ops
