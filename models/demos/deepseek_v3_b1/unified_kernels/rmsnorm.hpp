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
#include "compute_kernel_api.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/rsqrt.h"
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
//     - Reserves: scalars_cb (1 tile: reduction scalar)
//     - Pushes: scalars_cb (1 tile)
//     Note: input_cb and gamma_cb setup done externally via setup_sharded_buffer
//   BRISC: No-op (next op waits on output if needed)
//   TRISC (Compute):
//     - Waits: input_cb (num_tiles), scalars_cb (1), gamma_cb (num_tiles)
//     - Reserves: interm_cb (num_tiles), output_cb (num_tiles)
//     - Pushes: output_cb (num_tiles)
//     - Pops: input_cb (num_tiles) if pop_input=true
//     - Pops: interm_cb (used internally)
// ============================================================================
struct RMSNorm {
    // ========================================================================
    // Compile-time args structs - only what MUST be compile-time
    // (used as template parameters or in constexpr expressions)
    // ========================================================================

    // Reader CTArgs: num_faces (used as template param in generate_reduce_scaler)
    template <uint32_t NumFaces>
    struct ReaderCTArgs {
        static constexpr uint32_t num_faces = NumFaces;
    };

    // Writer CTArgs: none needed
    struct WriterCTArgs {};

    // Compute CTArgs: fp32_acc, num_tiles, rsqrt_fast_approx (template params)
    template <bool FP32Acc, uint32_t NumTiles, bool RsqrtFastApprox>
    struct ComputeCTArgs {
        static constexpr bool fp32_acc = FP32Acc;
        static constexpr uint32_t num_tiles = NumTiles;
        static constexpr bool rsqrt_fast_approx = RsqrtFastApprox;
    };

    // ========================================================================
    // Runtime args structs - different layout per RISC
    // ========================================================================
    // Reader args (NCRISC): only scalars needed (input_cb and gamma_cb setup done externally)
    struct ReaderArgs {
        uint32_t scalars_cb;
        uint32_t scalar;
    };
    // Writer args (BRISC): none (BRISC is no-op)
    struct WriterArgs {};
    struct ComputeArgs {
        uint32_t input_cb;
        uint32_t scalars_cb;
        uint32_t interm_cb;
        uint32_t gamma_cb;
        uint32_t output_cb;
        uint32_t epsilon;
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
#if defined(COMPILE_FOR_NCRISC)
            // ================================================================
            // NCRISC (Reader) - ReaderConfigDescriptor compiles as NCRISC
            // ================================================================
            // Generate reduction scalar (1/sqrt(num_elements))
            generate_reduce_scaler<CTArgs::num_faces>(args.scalars_cb, args.scalar);
#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC (Compute)
            // ================================================================
            // Init block done only once
            binary_op_init_common(args.input_cb, args.input_cb, args.output_cb);
            cb_wait_front(args.scalars_cb, 1);
            cb_wait_front(args.gamma_cb, CTArgs::num_tiles);  // we don't pop, only wait once and reuse

            compute_rmsnorm(args);
#endif
        }

#if defined(COMPILE_FOR_TRISC)
        void compute_rmsnorm(const ComputeArgs& args) {
            constexpr uint32_t num_tiles = CTArgs::num_tiles;
            {
                // Square the input
                mul_tiles_init(args.input_cb, args.input_cb);
                add_rsqrt_tile_init();
                cb_wait_front(args.input_cb, num_tiles);
                cb_reserve_back(args.interm_cb, num_tiles);
                tile_regs_acquire();
                for (uint32_t i = 0; i < num_tiles; i++) {
                    mul_tiles(args.input_cb, args.input_cb, i, i, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile_block(0, args.interm_cb, num_tiles);
                cb_push_back(args.interm_cb, num_tiles);
                tile_regs_release();

                // Calculate the avg of the sum of the squares
                reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR, CTArgs::fp32_acc>(
                    args.interm_cb, args.scalars_cb, args.interm_cb);
                cb_wait_front(args.interm_cb, num_tiles);
                tile_regs_acquire();
                for (uint32_t i = 0; i < num_tiles; i++) {
                    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR, CTArgs::fp32_acc>(
                        args.interm_cb, args.scalars_cb, i, 0, num_tiles);
                }
                cb_pop_front(args.interm_cb, num_tiles);
                cb_pop_front(args.scalars_cb, 1);  // Pop scalar tiles
                reduce_uninit();
            }
            {
                add_rsqrt_tile<CTArgs::rsqrt_fast_approx, VectorMode::RC_custom, 1>(num_tiles, args.epsilon);
            }
            {
                // Multiply input by 1/RMS
                rmsnorm_mul_bcast_scalar_reuse_tiles_init<num_tiles>(args.input_cb);
                rmsnorm_mul_bcast_scalar_reuse_tiles<num_tiles>(args.input_cb, 0, num_tiles, 0);
                if constexpr (pop_input) {
                    cb_pop_front(args.input_cb, num_tiles);
                }
            }
            {
                // Multiply by the weight
                cb_reserve_back(args.output_cb, num_tiles);
                binary_dest_reuse_tiles_init<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(args.gamma_cb);
                for (uint32_t i = 0; i < num_tiles; i++) {
                    binary_dest_reuse_tiles<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(args.gamma_cb, i, i);
                }

                tile_regs_commit();
                tile_regs_wait();
                pack_tile_block(0, args.output_cb, num_tiles);
                cb_push_back(args.output_cb, num_tiles);
                tile_regs_release();
            }
        }
#endif
    };  // class Op

};  // struct RMSNorm

}  // namespace deepseek_b1_ops
