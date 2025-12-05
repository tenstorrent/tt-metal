// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC)
#include "dataflow_api.h"
#elif defined(COMPILE_FOR_NCRISC)
#include "dataflow_api.h"
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
// ============================================================================
struct RMSNorm {
    // ========================================================================
    // Compile-time args structs - different layout per RISC
    // ========================================================================

    // Reader CTArgs: [input_cb, scalars_cb, gamma_cb, num_tiles, tiny_tile]
    template <uint32_t InputCB, uint32_t ScalarsCB, uint32_t GammaCB, uint32_t NumTiles, bool TinyTile>
    struct ReaderCTArgs {
        static constexpr uint32_t input_cb = InputCB;
        static constexpr uint32_t scalars_cb = ScalarsCB;
        static constexpr uint32_t gamma_cb = GammaCB;
        static constexpr uint32_t num_tiles = NumTiles;
        static constexpr bool tiny_tile = TinyTile;
    };

    // Writer CTArgs: [output_cb, num_tiles]
    template <uint32_t OutputCB, uint32_t NumTiles>
    struct WriterCTArgs {
        static constexpr uint32_t output_cb = OutputCB;
        static constexpr uint32_t num_tiles = NumTiles;
    };

    // Compute CTArgs: [input_cb, scalars_cb, interm_cb, gamma_cb, output_cb, fp32_acc, num_tiles, epsilon_index,
    // scalar_index, pop_input]
    template <
        uint32_t InputCB,
        uint32_t ScalarsCB,
        uint32_t IntermCB,
        uint32_t GammaCB,
        uint32_t OutputCB,
        bool FP32Acc,
        uint32_t NumTiles,
        uint32_t EpsilonIndex,
        uint32_t ScalarIndex,
        bool PopInput = true>
    struct ComputeCTArgs {
        static constexpr uint32_t input_cb = InputCB;
        static constexpr uint32_t scalars_cb = ScalarsCB;
        static constexpr uint32_t interm_cb = IntermCB;
        static constexpr uint32_t gamma_cb = GammaCB;
        static constexpr uint32_t output_cb = OutputCB;
        static constexpr bool fp32_acc = FP32Acc;
        static constexpr uint32_t num_tiles = NumTiles;
        static constexpr uint32_t epsilon_index = EpsilonIndex;
        static constexpr uint32_t scalar_index = ScalarIndex;
        static constexpr bool pop_input = PopInput;
    };

    // ========================================================================
    // Op - the actual operation, templated on CTArgs
    // ========================================================================
    template <typename CTArgs>
    class Op {
    public:
        // ====================================================================
        // Phase-specific RTArgs
        // ====================================================================
        struct ReaderArgs {
            uint32_t epsilon;
            uint32_t scalar;
        };
        struct WriterArgs {};
        struct ComputeArgs {};

        using RTArgs = SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

        void operator()(const RTArgs& args = {}) { impl(args); }

    private:
        void impl([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_NCRISC)
            // ================================================================
            // NCRISC (Reader) - ReaderConfigDescriptor compiles as NCRISC
            // ================================================================
            // Generate both scalar tiles in scalars_cb
            // Tile 0: epsilon
            // Tile 1: reduction scalar (1/sqrt(num_elements))
            cb_reserve_back(CTArgs::scalars_cb, 2);
            volatile tt_l1_ptr uint16_t* epsilon_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(CTArgs::scalars_cb));
            epsilon_ptr[0] = args.epsilon;
            cb_push_back(CTArgs::scalars_cb, 1);

            generate_reduce_scaler<CTArgs::tiny_tile>(CTArgs::scalars_cb, args.scalar);
            cb_push_back(CTArgs::scalars_cb, 1);

            // Signal that input and gamma buffers are ready (backed by L1 shards)
            cb_reserve_back(CTArgs::input_cb, CTArgs::num_tiles);
            cb_push_back(CTArgs::input_cb, CTArgs::num_tiles);
            cb_reserve_back(CTArgs::gamma_cb, CTArgs::num_tiles);
            cb_push_back(CTArgs::gamma_cb, CTArgs::num_tiles);
#elif defined(COMPILE_FOR_BRISC)
            // ================================================================
            // BRISC (Writer) - WriterConfigDescriptor compiles as BRISC
            // ================================================================
            // Wait for all output tiles to be available in CB
            // Note: output_cb is backed by sharded tensor, data will be written directly to L1
            cb_wait_front(CTArgs::output_cb, CTArgs::num_tiles);
#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC (Compute)
            // ================================================================
            // Init block done only once
            binary_op_init_common(CTArgs::input_cb, CTArgs::input_cb, CTArgs::output_cb);
            cb_wait_front(CTArgs::scalars_cb, 2);
            cb_wait_front(CTArgs::gamma_cb, CTArgs::num_tiles);  // we don't pop, only wait once and reuse
            rsqrt_tile_init();                                   // this is the only sfpu op we use, so we init once

            compute_rmsnorm();
#endif
        }

#if defined(COMPILE_FOR_TRISC)
        void compute_rmsnorm() {
            // TODO: #32998: Fuse this without having to spill output of square to interm cb
            {
                // Square the input
                mul_tiles_init(CTArgs::input_cb, CTArgs::input_cb);
                cb_wait_front(CTArgs::input_cb, CTArgs::num_tiles);
                cb_reserve_back(CTArgs::interm_cb, CTArgs::num_tiles + 1);  // Plus 1 for the RMS tile
                tile_regs_acquire();
                for (uint32_t i = 0; i < CTArgs::num_tiles; i++) {
                    mul_tiles(CTArgs::input_cb, CTArgs::input_cb, i, i, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile_block(0, CTArgs::interm_cb, CTArgs::num_tiles);
                tile_regs_release();
                cb_push_back(CTArgs::interm_cb, CTArgs::num_tiles);

                // Calculate the avg of the sum of the squares
                cb_wait_front(CTArgs::interm_cb, CTArgs::num_tiles);
                reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR, CTArgs::fp32_acc>(
                    CTArgs::interm_cb, CTArgs::scalars_cb, CTArgs::interm_cb);
                tile_regs_acquire();
                // TODO: #32998: Instead of accumulating to index 0, accumulate to num_tiles + 1 once bcast reuse is
                // supported
                for (uint32_t i = 0; i < CTArgs::num_tiles; i++) {
                    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR, CTArgs::fp32_acc>(
                        CTArgs::interm_cb, CTArgs::scalars_cb, i, CTArgs::scalar_index, 0);
                }
            }
            // TODO: #32998: Avoid having to spill 1/RMS to interm cb
            {
                // Add epsilon
                binary_dest_reuse_tiles_init<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CTArgs::scalars_cb);
                binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                    CTArgs::scalars_cb, CTArgs::epsilon_index, 0);
                // Calculate the 1/RMS
                // TODO: #32998: Use index num_tiles + 1 once bcast reuse is supported
                rsqrt_tile<false, true>(0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, CTArgs::interm_cb);
                tile_regs_release();
                reduce_uninit();
                cb_pop_front(CTArgs::interm_cb, CTArgs::num_tiles);
                cb_push_back(CTArgs::interm_cb, 1);  // 1/RMS tile should now be index 0
            }
            {
                // Multiply input by 1/RMS
                cb_wait_front(CTArgs::interm_cb, 1);
                cb_reserve_back(CTArgs::output_cb, CTArgs::num_tiles);
                mul_tiles_bcast_scalar_init_short(CTArgs::input_cb, CTArgs::interm_cb);
                tile_regs_acquire();
                for (uint32_t i = 0; i < CTArgs::num_tiles; i++) {
                    // TODO: #32998: Once we have bcast reuse, we will use input_cb index i, reuse dst index num_tiles +
                    // 1, output dst index i
                    mul_tiles_bcast_scalar(CTArgs::input_cb, CTArgs::interm_cb, i, 0, i);
                }
                if constexpr (CTArgs::pop_input) {
                    cb_pop_front(CTArgs::input_cb, CTArgs::num_tiles);
                }
            }
            {
                // Multiply by the weight
                binary_dest_reuse_tiles_init<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CTArgs::gamma_cb);
                for (uint32_t i = 0; i < CTArgs::num_tiles; i++) {
                    binary_dest_reuse_tiles<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CTArgs::gamma_cb, i, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile_block(0, CTArgs::output_cb, CTArgs::num_tiles);
                tile_regs_release();
                cb_pop_front(CTArgs::interm_cb, 1);
                cb_push_back(CTArgs::output_cb, CTArgs::num_tiles);
            }
        }
#endif
    };  // class Op

};  // struct RMSNorm

}  // namespace deepseek_b1_ops
