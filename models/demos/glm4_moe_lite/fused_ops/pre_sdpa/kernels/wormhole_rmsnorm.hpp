// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Wormhole-compatible RMSNorm for Pre-SDPA unified kernel.
// Replaces deepseek_v3_b1/unified_kernels/rmsnorm.hpp which uses
// Blackhole-only LLK APIs (mul_reduce_scalar, add_rsqrt_tile, rmsnorm_mul_bcast_scalar).
//
// Implementation strategy:
// - Uses output_cb as scratch for x^2 tiles (phase 1), pops, then reuses for final output
// - Uses scratch_cb for scaler/eps/rsqrt intermediate tiles
// - NCRISC fills scaler tile and eps tile into scratch_cb before Op runs
// - TRISC performs: x^2 -> reduce_scalar -> add_eps -> rsqrt -> x*rsqrt*gamma -> output
//
// Key difference from Blackhole version:
// - Uses REDUCE_SCALAR (not experimental/mul_reduce_scalar) for sum(x^2)
// - Uses standard rsqrt_tile (not custom add_rsqrt)
// - Uses mul_tiles_bcast_scalar (not rmsnorm_mul_bcast_scalar_reuse_tiles)
// - Uses binary_dest_reuse_tiles (available on both WH and BH) for gamma multiply
//
// CRITICAL: Must use REDUCE_SCALAR (not REDUCE_ROW) because RMSNorm computes
// the mean across ALL elements, not per-row. REDUCE_ROW would give wrong results
// for tiles with height > 1 (e.g., 16x32 or 32x32 tiles).

#pragma once

#include "../../../../deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_SCALAR
#include "api/compute/compute_kernel_api.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#endif

namespace deepseek_b1_ops {

// ============================================================================
// Wormhole-compatible RMSNorm micro-op
//
// Same external API as the Blackhole version (same struct names, same template
// parameters) but uses Wormhole-compatible compute primitives internally.
//
// CB States:
//   NCRISC (Reader):
//     - Fills scratch_cb with 2 tiles: scaler (tile 0) and epsilon (tile 1)
//     - Done via fill_rmsnorm_scratch_tile() calls in the kernel main
//   BRISC: No-op
//   TRISC (Compute):
//     - Waits: input_cb (num_tiles), gamma_cb (num_tiles)
//     - Uses: output_cb as scratch for x^2, scratch_cb for scaler/eps/rsqrt
//     - Pushes: output_cb (num_tiles)
//     - Pops: input_cb (num_tiles) if pop_input=true
// ============================================================================
struct RMSNorm {
    struct ReaderCTArgs {};
    struct WriterCTArgs {};

    template <bool FP32Acc, uint32_t NumTiles, bool RsqrtFastApprox>
    struct ComputeCTArgs {
        static constexpr bool fp32_acc = FP32Acc;
        static constexpr uint32_t num_tiles = NumTiles;
        static constexpr bool rsqrt_fast_approx = RsqrtFastApprox;
    };

    struct ReaderArgs {};
    struct WriterArgs {};
    struct ComputeArgs {
        uint32_t input_cb;
        uint32_t gamma_cb;
        uint32_t output_cb;
        uint32_t epsilon;       // unused on Wormhole (eps is in scratch_cb tile)
        float scalar;           // unused on Wormhole (scaler is in scratch_cb tile)
        uint32_t scratch_cb;    // Intermediate CB: tile 0 = scaler, tile 1 = eps
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
            compute_rmsnorm(args);
#endif
        }

#if defined(COMPILE_FOR_TRISC)
        void compute_rmsnorm(const ComputeArgs& args) {
            constexpr uint32_t num_tiles = CTArgs::num_tiles;
            constexpr uint32_t dst0 = 0;

            // DEBUG: rmsnorm_debug_phase controls how far compute progresses
            // 0 = skip all compute (test NCRISC setup only)
            // 1 = x^2 only (Phase 1)
            // 2 = x^2 + reduce (Phase 1+2)
            // 3 = x^2 + reduce + add_eps+rsqrt (Phase 1+2+3)
            // 5 = full RMSNorm (all phases)
            constexpr uint32_t rmsnorm_debug_phase = 5;  // CHANGE THIS TO TEST
            if constexpr (rmsnorm_debug_phase == 0) {
                return;
            }

            // Init common binary op state
            binary_op_init_common(args.input_cb, args.input_cb, args.output_cb);

            // Wait for persistent/pre-filled buffers
            cb_wait_front(args.gamma_cb, num_tiles);
            cb_wait_front(args.input_cb, num_tiles);

            // ============================================================
            // Phase 1: x^2 -> output_cb (used as temporary scratch)
            // ============================================================
            reconfig_data_format(args.input_cb, args.input_cb);
            pack_reconfig_data_format(args.output_cb);
            mul_tiles_init(args.input_cb, args.input_cb);
            cb_reserve_back(args.output_cb, num_tiles);
            for (uint32_t i = 0; i < num_tiles; i++) {
                tile_regs_acquire();
                mul_tiles(args.input_cb, args.input_cb, i, i, dst0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(dst0, args.output_cb);
                tile_regs_release();
            }
            cb_push_back(args.output_cb, num_tiles);

            // ============================================================
            // Phase 2: reduce(x^2) * scaler -> scalar value
            // Wait for scaler tile in scratch_cb (filled by NCRISC)
            // reduce_tile<SUM, REDUCE_SCALAR> reduces entire tile to [0,0]
            // and accumulates across all tiles, multiplied by scaler value.
            // Result: DST[0][0,0] = scaler * sum(x^2)
            // ============================================================
            cb_wait_front(args.output_cb, num_tiles);  // x^2 tiles
            cb_wait_front(args.scratch_cb, 1);         // scaler tile

            reconfig_data_format(args.output_cb, args.scratch_cb);
            pack_reconfig_data_format(args.scratch_cb);
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR, CTArgs::fp32_acc>(
                args.output_cb, args.scratch_cb, args.scratch_cb);
            tile_regs_acquire();
            for (uint32_t i = 0; i < num_tiles; i++) {
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR, CTArgs::fp32_acc>(
                    args.output_cb, args.scratch_cb, i, 0, dst0);
            }
            tile_regs_commit();

            // Pop x^2 scratch and scaler tile
            cb_pop_front(args.output_cb, num_tiles);
            cb_pop_front(args.scratch_cb, 1);

            // Pack variance scalar to scratch_cb
            tile_regs_wait();
            cb_reserve_back(args.scratch_cb, 1);
            pack_tile(dst0, args.scratch_cb);
            tile_regs_release();
            cb_push_back(args.scratch_cb, 1);
            reduce_uninit<CTArgs::fp32_acc>(args.scratch_cb);

            // ============================================================
            // Phase 3: rsqrt(variance + eps)
            // scratch_cb now has: [variance] (just packed)
            // NCRISC pushed eps as tile 1, which is now next in FIFO after
            // we popped the scaler and pushed variance.
            // So scratch_cb FIFO: [variance, eps]
            // We wait for 2 tiles, add them (scalar bcast), then rsqrt.
            // ============================================================
            cb_wait_front(args.scratch_cb, 2);  // variance + eps

            reconfig_data_format(args.scratch_cb, args.scratch_cb);
            pack_reconfig_data_format(args.scratch_cb);
            tile_regs_acquire();
            // Add eps to variance using scalar broadcast
            // (eps tile has value in [0,0], variance has value in [0,0])
            add_bcast_scalar_init_short(args.scratch_cb, args.scratch_cb);
            add_tiles_bcast_scalar(args.scratch_cb, args.scratch_cb, 0, 1, dst0);
            rsqrt_tile_init();
            rsqrt_tile<false, CTArgs::rsqrt_fast_approx>(dst0);
            tile_regs_commit();

            // Pop variance and eps, pack rsqrt result
            cb_pop_front(args.scratch_cb, 2);
            tile_regs_wait();
            cb_reserve_back(args.scratch_cb, 1);
            pack_tile(dst0, args.scratch_cb);
            tile_regs_release();
            cb_push_back(args.scratch_cb, 1);

            // ============================================================
            // Phase 4+5 FUSED: (x * rsqrt) * gamma -> output_cb
            // Uses scalar broadcast for 1/RMS multiply, then dest_reuse
            // for gamma multiply. Both work on Wormhole.
            // ============================================================
            cb_wait_front(args.scratch_cb, 1);  // rsqrt scalar
            cb_reserve_back(args.output_cb, num_tiles);
            for (uint32_t i = 0; i < num_tiles; i++) {
                tile_regs_acquire();
                // Step A: x * rsqrt(var+eps) via scalar broadcast
                reconfig_data_format(args.input_cb, args.scratch_cb);
                pack_reconfig_data_format(args.output_cb);
                mul_tiles_bcast_scalar_init_short(args.input_cb, args.scratch_cb);
                mul_tiles_bcast_scalar(args.input_cb, args.scratch_cb, i, 0, dst0);
                // Step B: dest * gamma via dest reuse (available on Wormhole)
                binary_dest_reuse_tiles_init<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(args.gamma_cb);
                binary_dest_reuse_tiles<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(args.gamma_cb, i, dst0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(dst0, args.output_cb);
                tile_regs_release();
            }
            cb_push_back(args.output_cb, num_tiles);
            cb_pop_front(args.scratch_cb, 1);  // Pop rsqrt scalar
            if constexpr (pop_input) {
                cb_pop_front(args.input_cb, num_tiles);
            }
        }
#endif
    };  // class Op

};  // struct RMSNorm

}  // namespace deepseek_b1_ops
