// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#elif defined(COMPILE_FOR_TRISC)
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_SCALAR
#include "compute_kernel_api.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/rsqrt.h"
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
//     - Reserves: scalars_cb (2 tiles: epsilon + reduction scalar)
//     - Pushes: scalars_cb (2 tiles)
//     Note: input_cb and gamma_cb setup done externally via setup_sharded_buffer
//   BRISC: No-op (next op waits on output if needed)
//   TRISC (Compute):
//     - Waits: input_cb (num_tiles), scalars_cb (2), gamma_cb (num_tiles)
//     - Reserves: interm_cb (num_tiles+1), output_cb (num_tiles)
//     - Pushes: output_cb (num_tiles)
//     - Pops: input_cb (num_tiles) if pop_input=true
//     - Pops: scalars_cb (2) always
//     - Pops: interm_cb (used internally)
// ============================================================================
struct RMSNorm {
    // ========================================================================
    // Compile-time args structs - only what MUST be compile-time
    // (used as template parameters or in constexpr expressions)
    // ========================================================================

    // Reader CTArgs: only tiny_tile (used as template param in generate_reduce_scaler)
    template <bool TinyTile>
    struct ReaderCTArgs {
        static constexpr bool tiny_tile = TinyTile;
    };

    // Writer CTArgs: none needed
    struct WriterCTArgs {};

    // Compute CTArgs: fp32_acc (template param)
    template <bool FP32Acc>
    struct ComputeCTArgs {
        static constexpr bool fp32_acc = FP32Acc;
    };

    // ========================================================================
    // Runtime args structs - different layout per RISC
    // ========================================================================
    // Reader args (NCRISC): only scalars needed (input_cb and gamma_cb setup done externally)
    struct ReaderArgs {
        uint32_t scalars_cb;
        uint32_t epsilon;
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
        uint32_t num_tiles;
        uint32_t epsilon_index;
        uint32_t scalar_index;
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
            // Generate both scalar tiles in scalars_cb
            // Tile 0: epsilon
            // Tile 1: reduction scalar (1/sqrt(num_elements))
            cb_reserve_back(args.scalars_cb, 1);
            volatile tt_l1_ptr uint16_t* epsilon_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(args.scalars_cb));
            epsilon_ptr[0] = args.epsilon;
            cb_push_back(args.scalars_cb, 1);

            generate_reduce_scaler<CTArgs::tiny_tile>(args.scalars_cb, args.scalar);
#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC (Compute)
            // ================================================================
            // Init block done only once
            binary_op_init_common(args.input_cb, args.input_cb, args.output_cb);
            cb_wait_front(args.scalars_cb, 2);
            cb_wait_front(args.gamma_cb, args.num_tiles);  // we don't pop, only wait once and reuse
            rsqrt_tile_init();                             // this is the only sfpu op we use, so we init once

            compute_rmsnorm(args);
#endif
        }

#if defined(COMPILE_FOR_TRISC)
        void compute_rmsnorm(const ComputeArgs& args) {
            // TODO: #32998: Fuse this without having to spill output of square to interm cb
            {
                // Square the input
                mul_tiles_init(args.input_cb, args.input_cb);
                cb_wait_front(args.input_cb, args.num_tiles);
                cb_reserve_back(args.interm_cb, args.num_tiles + 1);  // Plus 1 for the RMS tile
                tile_regs_acquire();
                for (uint32_t i = 0; i < args.num_tiles; i++) {
                    mul_tiles(args.input_cb, args.input_cb, i, i, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile_block(0, args.interm_cb, args.num_tiles);
                tile_regs_release();
                cb_push_back(args.interm_cb, args.num_tiles);

                // Calculate the avg of the sum of the squares
                cb_wait_front(args.interm_cb, args.num_tiles);
                reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR, CTArgs::fp32_acc>(
                    args.interm_cb, args.scalars_cb, args.interm_cb);
                tile_regs_acquire();
                // TODO: #32998: Instead of accumulating to index 0, accumulate to num_tiles + 1 once bcast reuse is
                // supported
                for (uint32_t i = 0; i < args.num_tiles; i++) {
                    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR, CTArgs::fp32_acc>(
                        args.interm_cb, args.scalars_cb, i, args.scalar_index, 0);
                }
            }
            // TODO: #32998: Avoid having to spill 1/RMS to interm cb
            {
                // Add epsilon
                binary_dest_reuse_tiles_init<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(args.scalars_cb);
                binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                    args.scalars_cb, args.epsilon_index, 0);
                // Calculate the 1/RMS
                // TODO: #32998: Use index num_tiles + 1 once bcast reuse is supported
                rsqrt_tile<false, true>(0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, args.interm_cb);
                tile_regs_release();
                reduce_uninit();
                cb_pop_front(args.scalars_cb, 2);  // Pop epsilon and scalar tiles
                cb_pop_front(args.interm_cb, args.num_tiles);
                cb_push_back(args.interm_cb, 1);  // 1/RMS tile should now be index 0
            }
            {
                // Multiply input by 1/RMS
                cb_wait_front(args.interm_cb, 1);
                cb_reserve_back(args.output_cb, args.num_tiles);
                mul_tiles_bcast_scalar_init_short(args.input_cb, args.interm_cb);
                tile_regs_acquire();
                for (uint32_t i = 0; i < args.num_tiles; i++) {
                    // TODO: #32998: Once we have bcast reuse, we will use input_cb index i, reuse dst index num_tiles +
                    // 1, output dst index i
                    mul_tiles_bcast_scalar(args.input_cb, args.interm_cb, i, 0, i);
                }
                if constexpr (pop_input) {
                    cb_pop_front(args.input_cb, args.num_tiles);
                }
            }
            {
                // Multiply by the weight
                binary_dest_reuse_tiles_init<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(args.gamma_cb);
                for (uint32_t i = 0; i < args.num_tiles; i++) {
                    binary_dest_reuse_tiles<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(args.gamma_cb, i, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile_block(0, args.output_cb, args.num_tiles);
                tile_regs_release();
                cb_pop_front(args.interm_cb, 1);

                cb_push_back(args.output_cb, args.num_tiles);
            }
        }
#endif
    };  // class Op

};  // struct RMSNorm

}  // namespace deepseek_b1_ops
