// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"
#include "kernel_utils.hpp"

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
#include "api/compute/experimental/pack_block.h"
#include "api/compute/tile_move_copy.h"
#include "../kernel_includes/tt_metal/include/compute_kernel_api/add_rsqrt.h"
#include "../kernel_includes/tt_metal/include/compute_kernel_api/rmsnorm.h"
#include "../kernel_includes/tt_metal/include/compute_kernel_api/eltwise_mul_scalar.h"
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
        uint32_t OutputCb,
        bool DoGamma = true>
    struct ComputeCTArgs {
        static constexpr bool fp32_acc = FP32Acc;
        static constexpr uint32_t num_tiles = NumTiles;
        static constexpr bool rsqrt_fast_approx = RsqrtFastApprox;
        static constexpr uint32_t input_cb = InputCb;
        static constexpr uint32_t gamma_cb = GammaCb;
        static constexpr uint32_t output_cb = OutputCb;
        static constexpr bool do_gamma = DoGamma;
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
        uint32_t gamma_address_override = 0;  // byte address; overrides gamma read ptr if > 0
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
            if constexpr (CTArgs::do_gamma) {
                // Init block done only once; we don't pop, only wait once and reuse
                if (args.gamma_address_override > 0) {
                    UNPACK(({ unified_kernels::override_cb_rd_ptr(CTArgs::gamma_cb, args.gamma_address_override); }));
                } else {
                    cb_wait_front(CTArgs::gamma_cb, CTArgs::num_tiles);
                }
            }

            compute_rmsnorm(args);
#endif
        }

#if defined(COMPILE_FOR_TRISC)
        void compute_rmsnorm(const ComputeArgs& args) {
            constexpr uint32_t num_tiles = CTArgs::num_tiles;
            reconfig_data_format<false, true>(CTArgs::input_cb, CTArgs::input_cb);
            pack_reconfig_data_format<true>(CTArgs::output_cb);
            pack_block_contiguous_init(CTArgs::output_cb);
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
            }
            {
                cb_reserve_back(CTArgs::output_cb, num_tiles);
                if constexpr (CTArgs::do_gamma) {
                    binary_dest_reuse_tiles_init<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CTArgs::gamma_cb);
                    for (uint32_t i = 0; i < num_tiles; i++) {
                        binary_dest_reuse_tiles<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                            CTArgs::gamma_cb, i, i);
                    }
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_block_contiguous(0, CTArgs::output_cb, num_tiles);
                cb_push_back(CTArgs::output_cb, num_tiles);
                tile_regs_release();
            }
            if constexpr (pop_input) {
                cb_pop_front(CTArgs::input_cb, num_tiles);
            }
        }
#endif
    };  // class Op

};  // struct RMSNorm

// ============================================================================
// RMSInverse micro-op (front half of RMSNorm)
//
// Computes: output_cb = 1 / RMS(input) = rsqrt(mean(input^2) + epsilon)
// Output is a single scalar tile per invocation. Unlike RMSNorm, the inverse
// is NOT applied to the input and gamma is not consulted — only the inverse
// RMS scalar is produced. Intended for paths that want to compute the RMS
// statistic on the sender side and consume / forward it later.
//
// CB States:
//   NCRISC (Reader): no-op
//   BRISC (Writer):  no-op
//   TRISC (Compute):
//     - Waits: input_cb (num_tiles)
//     - Reserves: output_cb (1 tile)
//     - Pushes: output_cb (1 tile)
//     - Pops: input_cb (num_tiles) if pop_input=true
// ============================================================================
struct RMSInverse {
    // Reader CTArgs: none needed
    struct ReaderCTArgs {};
    // Writer CTArgs: none needed
    struct WriterCTArgs {};

    // Compute CTArgs: fp32_acc, num_tiles, rsqrt_fast_approx, input_cb, output_cb
    template <bool FP32Acc, uint32_t NumTiles, bool RsqrtFastApprox, uint32_t InputCb, uint32_t OutputCb>
    struct ComputeCTArgs {
        static constexpr bool fp32_acc = FP32Acc;
        static constexpr uint32_t num_tiles = NumTiles;
        static constexpr bool rsqrt_fast_approx = RsqrtFastApprox;
        static constexpr uint32_t input_cb = InputCb;
        static constexpr uint32_t output_cb = OutputCb;
    };

    // Runtime args (Reader/Writer empty; Compute carries epsilon + scalar)
    struct ReaderArgs {};
    struct WriterArgs {};
    struct ComputeArgs {
        uint32_t epsilon;
        float scalar;
    };

    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

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
            compute_rms_inverse(args);
#endif
        }

#if defined(COMPILE_FOR_TRISC)
        void compute_rms_inverse(const ComputeArgs& args) {
            constexpr uint32_t num_tiles = CTArgs::num_tiles;
            reconfig_data_format<false, true>(CTArgs::input_cb, CTArgs::input_cb);
            pack_reconfig_data_format<true>(CTArgs::output_cb);
            pack_block_contiguous_init(CTArgs::output_cb);

            mul_reduce_scalar_init(CTArgs::input_cb, CTArgs::input_cb);
            add_rsqrt_tile_init();
            cb_wait_front(CTArgs::input_cb, num_tiles);
            tile_regs_acquire();
            mul_reduce_scalar_tile<PoolType::SUM>(CTArgs::input_cb, CTArgs::input_cb, num_tiles, args.scalar);
            mul_reduce_scalar_uninit();

            add_rsqrt_tile<CTArgs::rsqrt_fast_approx, VectorMode::RC_custom, 1>(0, args.epsilon);

            cb_reserve_back(CTArgs::output_cb, 1);
            tile_regs_commit();
            tile_regs_wait();
            pack_block_contiguous(0, CTArgs::output_cb, 1);
            cb_push_back(CTArgs::output_cb, 1);
            tile_regs_release();

            if constexpr (pop_input) {
                cb_pop_front(CTArgs::input_cb, num_tiles);
            }
        }
#endif
    };  // class Op

};  // struct RMSInverse

// ============================================================================
// RMSApply micro-op
//
// Computes: output_cb[i] = input_cb[i] * scalar_cb[0] (broadcast from [0,0])
// for each tile i in [0, num_tiles).
//
// Pairs with RMSInverse: the sender produces 1/RMS via RMSInverse, mcasts the
// scalar tile, and consumers run RMSApply to scale a downstream tensor by it.
// Supports in-place use: pass the same CB id for input_cb and output_cb to
// modify the input CB's tiles in place (the op pops the input tiles, then
// reserves+packs+pushes the same number of tiles back to that CB).
//
// CB States:
//   NCRISC (Reader): no-op
//   BRISC (Writer):  no-op
//   TRISC (Compute):
//     - Waits: input_cb (num_tiles), scalar_cb (1)
//     - Pops: input_cb (num_tiles)
//     - Reserves: output_cb (num_tiles)
//     - Pushes: output_cb (num_tiles)
//     - Pops: scalar_cb (1) if pop_scalar=true
// ============================================================================
struct RMSApply {
    // Reader CTArgs: none needed
    struct ReaderCTArgs {};
    // Writer CTArgs: none needed
    struct WriterCTArgs {};

    // Compute CTArgs: fp32_acc, num_tiles, input_cb, scalar_cb, output_cb
    template <bool FP32Acc, uint32_t NumTiles, uint32_t InputCb, uint32_t ScalarCb, uint32_t OutputCb>
    struct ComputeCTArgs {
        static constexpr bool fp32_acc = FP32Acc;
        static constexpr uint32_t num_tiles = NumTiles;
        static constexpr uint32_t input_cb = InputCb;
        static constexpr uint32_t scalar_cb = ScalarCb;
        static constexpr uint32_t output_cb = OutputCb;
    };

    struct ReaderArgs {};
    struct WriterArgs {};
    struct ComputeArgs {};

    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

    template <typename CTArgs, bool IsActiveCore, bool pop_scalar>
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
            constexpr uint32_t num_tiles = CTArgs::num_tiles;

            cb_wait_front(CTArgs::input_cb, num_tiles);
            cb_wait_front(CTArgs::scalar_cb, 1);

            reconfig_data_format<false, true>(CTArgs::input_cb, CTArgs::scalar_cb);
            pack_reconfig_data_format<true>(CTArgs::output_cb);
            deepseek_mul_tiles_bcast_scalar_init_short(CTArgs::input_cb, CTArgs::scalar_cb);

            tile_regs_acquire();
            for (uint32_t i = 0; i < num_tiles; i++) {
                deepseek_mul_tiles_bcast_scalar<CTArgs::fp32_acc>(CTArgs::input_cb, CTArgs::scalar_cb, i, 0, i);
            }
            tile_regs_commit();

            // Pop input first so an in-place output_cb (input_cb == output_cb) can
            // reserve back the same slots without blocking on a full CB.
            cb_pop_front(CTArgs::input_cb, num_tiles);
            cb_reserve_back(CTArgs::output_cb, num_tiles);
            tile_regs_wait();
            for (uint32_t i = 0; i < num_tiles; i++) {
                pack_tile(i, CTArgs::output_cb);
            }
            cb_push_back(CTArgs::output_cb, num_tiles);
            tile_regs_release();

            if constexpr (pop_scalar) {
                cb_pop_front(CTArgs::scalar_cb, 1);
            }
#endif
        }
    };  // class Op

};  // struct RMSApply

}  // namespace deepseek_b1_ops
