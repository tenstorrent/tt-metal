// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Wormhole-compatible RMSNorm micro-op for GLM KV Cache Branch.
//
// ALL circular buffers use TILE_1x32 (1 row x 32 cols) format.
// Uses REDUCE_ROW (not REDUCE_SCALAR) — on a 1-row tile this naturally
// produces a scalar (sum of the 32 elements in the single row).
// The scaler tile (1x32 with 1/N in all 32 positions) scales during reduce.
// After accumulating across all tiles, dest[0] = mean(x^2).
//
// Phase 1: x^2 (mul_tiles) -> cb_x2 (num_tiles)
// Phase 2: reduce_tile<SUM, REDUCE_ROW> over all cb_x2 tiles -> cb_var (1 tile)
// Phase 3: add eps + rsqrt -> cb_var (1 tile)
// Phase 4+5: mul_bcast_cols(input, rsqrt) * gamma -> output_cb (num_tiles)
//
// All intermediate CBs (cb_x2, cb_var, cb_scaler, cb_eps) are TILE_1x32.

#pragma once

#include "../../../../deepseek_v3_b1/unified_kernels/kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#include "api/compute/compute_kernel_api.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#endif

namespace glm4_rmsnorm {

// ============================================================================
// Wormhole RMSNorm micro-op (TILE_1x32 throughout)
//
// Computes: output = (input / RMS(input)) * gamma
// Where RMS(x) = sqrt(mean(x^2) + epsilon)
//
// CB States:
//   NCRISC (Reader):
//     - input_cb and gamma_cb setup done externally via setup_sharded_buffer
//     - Fills cb_scaler (1x32 tile with 1/N) and cb_eps (1x32 tile with eps)
//   BRISC: No-op
//   TRISC (Compute):
//     - Waits: input_cb (num_tiles), gamma_cb (num_tiles), cb_scaler (1), cb_eps (1)
//     - Uses: cb_x2 (num_tiles intermediate), cb_var (1 tile intermediate)
//     - Pushes: output_cb (num_tiles)
//     - Pops: input_cb (num_tiles) if pop_input=true
// ============================================================================
struct RMSNorm {
    // ========================================================================
    // Compile-time args structs
    // ========================================================================
    struct ReaderCTArgs {};
    struct WriterCTArgs {};

    template <bool FP32Acc, uint32_t NumTiles, bool RsqrtFastApprox>
    struct ComputeCTArgs {
        static constexpr bool fp32_acc = FP32Acc;
        static constexpr uint32_t num_tiles = NumTiles;
        static constexpr bool rsqrt_fast_approx = RsqrtFastApprox;
    };

    // ========================================================================
    // Runtime args structs
    // ========================================================================
    struct ReaderArgs {
        uint32_t cb_scaler;
        uint32_t cb_eps;
        uint32_t scaler_packed;  // bf16 1/N packed as uint16
        uint32_t eps_packed;     // bf16 epsilon packed as uint16
    };
    struct WriterArgs {};
    struct ComputeArgs {
        uint32_t input_cb;
        uint32_t gamma_cb;
        uint32_t output_cb;
        uint32_t cb_x2;      // intermediate: x^2 tiles (num_tiles)
        uint32_t cb_var;     // intermediate: reduced variance + rsqrt result (1 tile)
        uint32_t cb_scaler;  // pre-filled reduce scaler tile (1/N in all 32 cols)
        uint32_t cb_eps;     // pre-filled epsilon tile (eps in all 32 cols)
    };

    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

    // ========================================================================
    // Op
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
            // NCRISC: Fill scaler and epsilon CBs with TILE_1x32 tiles
            // ================================================================
            fill_1x32_tile(args.cb_scaler, args.scaler_packed);
            fill_1x32_tile(args.cb_eps, args.eps_packed);

#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC (Compute)
            // ================================================================
            compute_rmsnorm(args);
#endif
        }

#if defined(COMPILE_FOR_NCRISC)
        // Fill a TILE_1x32 CB with a uniform bf16 value in all 32 positions.
        // For REDUCE_ROW scaler: value = bf16(1/N) in all 32 cols of the single row.
        // For epsilon: value = bf16(eps) in all 32 cols.
        static void fill_1x32_tile(uint32_t cb_id, uint32_t bf16_val_packed) {
            cb_reserve_back(cb_id, 1);
            uint32_t write_addr = get_write_ptr(cb_id);

            // Zero the tile first using MEM_ZEROS
            uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
            noc_async_read_one_packet_set_state(zeros_noc_addr, MEM_ZEROS_SIZE);
            noc_async_read_one_packet_with_state(zeros_noc_addr, write_addr);
            noc_async_read_barrier();

            // Fill all 32 bf16 positions with the value
            volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(write_addr);
            uint16_t val = static_cast<uint16_t>(bf16_val_packed);
            for (uint32_t j = 0; j < 32; ++j) {
                ptr[j] = val;
            }

            cb_push_back(cb_id, 1);
        }
#endif

#if defined(COMPILE_FOR_TRISC)
        void compute_rmsnorm(const ComputeArgs& args) {
            constexpr uint32_t num_tiles = CTArgs::num_tiles;
            constexpr uint32_t dst0 = 0;

            // Init common binary op state
            binary_op_init_common(args.input_cb, args.input_cb, args.cb_x2);

            // Wait for persistent/pre-filled buffers
            cb_wait_front(args.gamma_cb, num_tiles);
            cb_wait_front(args.input_cb, num_tiles);
            cb_wait_front(args.cb_scaler, 1);
            cb_wait_front(args.cb_eps, 1);

            // ============================================================
            // Phase 1: x^2 (element-wise square) -> cb_x2
            // ============================================================
            reconfig_data_format(args.input_cb, args.input_cb);
            pack_reconfig_data_format(args.cb_x2);
            mul_tiles_init(args.input_cb, args.input_cb);
            cb_reserve_back(args.cb_x2, num_tiles);
            for (uint32_t i = 0; i < num_tiles; i++) {
                tile_regs_acquire();
                mul_tiles(args.input_cb, args.input_cb, i, i, dst0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(dst0, args.cb_x2);
                tile_regs_release();
            }
            cb_push_back(args.cb_x2, num_tiles);

            // ============================================================
            // Phase 2: sum(x^2) * scaler via REDUCE_ROW -> mean(x^2)
            // reduce_tile<SUM, REDUCE_ROW> on each 1x32 tile sums 32 elems.
            // The scaler (1/N) is applied per tile during reduction.
            // Accumulate across all tiles in dest[0].
            // ============================================================
            cb_wait_front(args.cb_x2, num_tiles);
            reconfig_data_format(args.cb_x2, args.cb_scaler);
            pack_reconfig_data_format(args.cb_var);
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(args.cb_x2, args.cb_scaler, args.cb_var);
            cb_reserve_back(args.cb_var, 1);
            tile_regs_acquire();
            for (uint32_t i = 0; i < num_tiles; i++) {
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(args.cb_x2, args.cb_scaler, i, 0, dst0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, args.cb_var);
            tile_regs_release();
            cb_push_back(args.cb_var, 1);
            reduce_uninit();
            cb_pop_front(args.cb_x2, num_tiles);

            // ============================================================
            // Phase 3: rsqrt(mean(x^2) + eps)
            // add_bcast_cols: broadcasts eps col 0 across all cols, adds
            // rsqrt_tile: 1/sqrt(dest) -> dest
            // ============================================================
            cb_wait_front(args.cb_var, 1);
            reconfig_data_format(args.cb_var, args.cb_eps);
            pack_reconfig_data_format(args.cb_var);
            tile_regs_acquire();
            add_bcast_cols_init_short(args.cb_var, args.cb_eps);
            add_tiles_bcast_cols(args.cb_var, args.cb_eps, 0, 0, dst0);
            rsqrt_tile_init<CTArgs::rsqrt_fast_approx>();
            rsqrt_tile<CTArgs::rsqrt_fast_approx>(dst0);
            tile_regs_commit();
            tile_regs_wait();
            cb_pop_front(args.cb_var, 1);
            cb_reserve_back(args.cb_var, 1);
            pack_tile(dst0, args.cb_var);
            tile_regs_release();
            cb_push_back(args.cb_var, 1);

            // ============================================================
            // Phase 4+5 FUSED: (x * rsqrt_scalar) * gamma -> output_cb
            // mul_bcast_cols: broadcasts col 0 of cb_var (the rsqrt scalar)
            //   across all 32 columns of each input tile
            // binary_dest_reuse: dest * gamma -> output
            // ============================================================
            cb_wait_front(args.cb_var, 1);
            cb_reserve_back(args.output_cb, num_tiles);
            for (uint32_t i = 0; i < num_tiles; i++) {
                tile_regs_acquire();
                // Step A: x * rsqrt(var+eps) via column broadcast
                reconfig_data_format(args.input_cb, args.cb_var);
                pack_reconfig_data_format(args.output_cb);
                mul_bcast_cols_init_short(args.input_cb, args.cb_var);
                mul_tiles_bcast_cols(args.input_cb, args.cb_var, i, 0, dst0);
                // Step B: dest * gamma via dest reuse
                binary_dest_reuse_tiles_init<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(args.gamma_cb);
                binary_dest_reuse_tiles<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(args.gamma_cb, i, dst0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(dst0, args.output_cb);
                tile_regs_release();
            }
            cb_push_back(args.output_cb, num_tiles);
            if constexpr (pop_input) {
                cb_pop_front(args.input_cb, num_tiles);
            }
            cb_pop_front(args.cb_var, 1);
        }
#endif
    };  // class Op

};  // struct RMSNorm

}  // namespace glm4_rmsnorm
