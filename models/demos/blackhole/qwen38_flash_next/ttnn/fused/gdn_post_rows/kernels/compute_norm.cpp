// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// ``post_norm`` compute: ``ttnn.rms_norm(o16 [1, 12, T, 128] bf16, weight=norm, epsilon=1e-6)`` followed by
// ``ttnn.multiply(normalized, sigmoid_bf16)``, call for call, on the four column tiles of one (value head, tile row).
//
// The norm is layernorm.cpp's RMSNORM + FUSE_GAMMA path at the op's default compute config (HiFi4, math_approx_mode
// true, fp32_dest_acc_en false): a 16-bit DEST, bf16 intermediates, Wt = 4 in one block (the op reserves 8 CB slots
// per block; reserving 4 changes no operand).  The kernel compiles APPROX = false for the gate (binary_ng's Precise
// config), so the norm's two SFPU calls go through the LLK macros with APPROX = true spelled out.
//
// The gate is binary_ng's eltwise_binary_sfpu_no_bcast.cpp: copy both operands to DST, ``mul_binary_tile``, one pack.
//
// CBs: cb_x (0, bf16, the o16 tiles), cb_sig (1), cb_scaler (2), cb_eps (3), cb_gamma (4), cb_xmm2 (6),
// cb_ex2 (7), cb_ex2pe (8), cb_fusion (9), cb_nrm (10), cb_out (16); cb_proj (5) and cb_hist (17) are the
// writer's history path and never reach the compute.  Runtime args: 0 units on this core.

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"

using namespace ckernel;

namespace {
constexpr uint32_t cb_x = 0, cb_sig = 1, cb_scaler = 2, cb_eps = 3, cb_gamma = 4;
constexpr uint32_t cb_xmm2 = 6, cb_ex2 = 7, cb_ex2pe = 8, cb_fusion = 9, cb_nrm = 10, cb_out = 16;
constexpr uint32_t head_tiles = 4;  // Wt: 128 / 32
constexpr uint32_t dst0 = 0, dst1 = 1;
// numeric.h's row_wise_mean epilogue: scale_dest(dst0, bit_cast<uint32_t>(1.0f / W)) with W = 128
constexpr uint32_t inverse_width_bits = 0x3C000000u;
}  // namespace

void kernel_main() {
    const uint32_t units = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_x, cb_x, cb_xmm2);
    CircularBuffer(cb_eps).wait_front(1);

    for (uint32_t unit = 0; unit < units; ++unit) {
        CircularBuffer(cb_x).wait_front(head_tiles);

        // x * x -> cb_xmm2 (bf16)
        {
            reconfig_data_format(cb_x, cb_x);
            pack_reconfig_data_format(cb_xmm2);
            mul_init(cb_x, cb_x);
            tile_regs_acquire();
            for (uint32_t i = 0; i < head_tiles; ++i) {
                mul_tiles(cb_x, cb_x, i, i, i);
            }
            tile_regs_commit();
            CircularBuffer(cb_xmm2).reserve_back(head_tiles);
            tile_regs_wait();
            for (uint32_t i = 0; i < head_tiles; ++i) {
                pack_tile(i, cb_xmm2);
            }
            tile_regs_release();
            CircularBuffer(cb_xmm2).push_back(head_tiles);
        }
        reconfig_data_format(cb_x, cb_xmm2, cb_x, cb_scaler);

        // mean(x * x): numeric::row_wise_mean<SUM, REDUCE_ROW, FLOAT32_REDUCTION = false, FullBlockWithPopPolicy>
        {
            CircularBuffer(cb_scaler).wait_front(1);
            reconfig_data_format(cb_xmm2, cb_scaler);
            tile_regs_acquire();
            reconfig_data_format(cb_scaler, cb_xmm2);  // REDUCE_ROW SUM swaps the operands: scaler SrcA, data SrcB
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_xmm2, cb_scaler, cb_ex2);
            CircularBuffer(cb_xmm2).wait_front(head_tiles);
            for (uint32_t j = 0; j < head_tiles; ++j) {
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_xmm2, cb_scaler, j, 0, dst0);
            }
            CircularBuffer(cb_xmm2).pop_front(head_tiles);
            // numeric.h's own order: accumulate_compute_loop ends with reduce_uninit + reconfig and the epilogue
            // (scale_dest) runs after it, still inside the same tile_regs_acquire
            reduce_uninit();
            reconfig_data_format(cb_xmm2, cb_scaler);
            // numeric.h's scale_dest = binop_with_scalar_tile_init() + mul_unary_tile(dst0, 1/W), with APPROX = true
            // in place of the kernel's: calculate_binop_with_scalar<true, MUL_UNARY, 8, false>.
            binop_with_scalar_tile_init();
            MATH(SFPU_UNARY_CALL(
                DST_SYNC_MODE,
                false /* is_fp32_dest_acc_en */,
                calculate_binop_with_scalar,
                (true /* APPROX */, MUL_UNARY, 8 /* ITERATIONS */, false /* is_fp32_dest_acc_en */),
                dst0,
                VectorMode::RC,
                inverse_width_bits));
            tile_regs_commit();
            tile_regs_wait();
            CircularBuffer(cb_ex2).reserve_back(1);
            pack_reconfig_data_format(cb_ex2);
            pack_tile(dst0, cb_ex2);
            tile_regs_release();
            CircularBuffer(cb_ex2).push_back(1);
        }

        // rsqrt(mean + eps); the eps tile is the fp32 bits of 1e-6 truncated to bf16 (generate_bcast_col_scalar)
        {
            CircularBuffer(cb_ex2).wait_front(1);
            reconfig_data_format(cb_ex2, cb_eps);
            tile_regs_acquire();
            add_init(cb_ex2, cb_eps);
            add_tiles(cb_ex2, cb_eps, 0, 0, dst0);
            // rsqrt_tile_init() / rsqrt_tile(dst0) as layernorm.cpp calls them, with APPROX = true in place of the
            // kernel's: rsqrt_init<true> and calculate_rsqrt<true, 8, false, false>, as math_approx_mode selects.
            MATH(SFPU_UNARY_INIT_FN(rsqrt, sfpu::rsqrt_init, (true /* APPROX */)));
            MATH(SFPU_UNARY_CALL(
                DST_SYNC_MODE,
                false /* is_fp32_dest_acc_en */,
                calculate_rsqrt,
                (true /* APPROX */, 8 /* ITERATIONS */, false /* is_fp32_dest_acc_en */, false /* FAST_APPROX */),
                dst0,
                VectorMode::RC));
            tile_regs_commit();
            CircularBuffer(cb_ex2).pop_front(1);
            CircularBuffer(cb_ex2pe).reserve_back(1);
            pack_reconfig_data_format(cb_ex2pe);
            tile_regs_wait();
            pack_tile(dst0, cb_ex2pe);
            tile_regs_release();
            CircularBuffer(cb_ex2pe).push_back(1);
        }

        // x * rsqrt (FPU column broadcast) -> cb_fusion, then * gamma (FPU row broadcast) -> cb_nrm
        {
            CircularBuffer(cb_ex2pe).wait_front(1);
            reconfig_data_format(cb_x, cb_ex2pe);
            pack_reconfig_data_format(cb_fusion);
            CircularBuffer(cb_fusion).reserve_back(head_tiles);
            reconfig_data_format_srca(cb_fusion, cb_x);
            tile_regs_acquire();
            mul_bcast_cols_init(cb_x, cb_ex2pe);
            for (uint32_t i = 0; i < head_tiles; ++i) {
                mul_tiles_bcast_cols(cb_x, cb_ex2pe, i, 0, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < head_tiles; ++i) {
                pack_tile(i, cb_fusion);
            }
            tile_regs_release();
            CircularBuffer(cb_fusion).push_back(head_tiles);
            reconfig_data_format_srca(cb_x, cb_fusion);
        }

        {
            pack_reconfig_data_format(cb_nrm);
            reconfig_data_format_srcb(cb_ex2pe, cb_gamma);
            CircularBuffer(cb_gamma).wait_front(head_tiles);
            CircularBuffer(cb_fusion).wait_front(head_tiles);
            tile_regs_acquire();
            mul_bcast_rows_init(cb_fusion, cb_gamma);
            for (uint32_t i = 0; i < head_tiles; ++i) {
                mul_tiles_bcast_rows(cb_fusion, cb_gamma, i, i, i);
            }
            tile_regs_commit();
            CircularBuffer(cb_fusion).pop_front(head_tiles);
            CircularBuffer(cb_nrm).reserve_back(head_tiles);
            tile_regs_wait();
            for (uint32_t i = 0; i < head_tiles; ++i) {
                pack_tile(i, cb_nrm);
            }
            tile_regs_release();
            CircularBuffer(cb_nrm).push_back(head_tiles);
            CircularBuffer(cb_ex2pe).pop_front(1);
            CircularBuffer(cb_x).pop_front(head_tiles);
        }

        // the gate: ttnn.multiply(normalized, sigmoid_bf16) = binary_ng's SFPU multiply, one tile per acquire
        {
            CircularBuffer(cb_nrm).wait_front(head_tiles);
            CircularBuffer(cb_sig).wait_front(head_tiles);
            CircularBuffer(cb_out).reserve_back(head_tiles);
            pack_reconfig_data_format(cb_out);
            for (uint32_t i = 0; i < head_tiles; ++i) {
                tile_regs_acquire();
                reconfig_data_format_srca(cb_sig, cb_nrm);
                copy_init(cb_nrm);
                copy_tile(cb_nrm, i, dst0);
                reconfig_data_format_srca(cb_nrm, cb_sig);
                copy_init(cb_sig);
                copy_tile(cb_sig, i, dst1);
                mul_binary_tile_init();
                mul_binary_tile(0, 1, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(dst0, cb_out);
                tile_regs_release();
            }
            CircularBuffer(cb_out).push_back(head_tiles);
            CircularBuffer(cb_nrm).pop_front(head_tiles);
            CircularBuffer(cb_sig).pop_front(head_tiles);
        }
    }

    // the three constant tiles are read once and never popped inside the loop (the op's own gamma / scaler rule)
    CircularBuffer(cb_gamma).pop_front(head_tiles);
    CircularBuffer(cb_scaler).pop_front(1);
    CircularBuffer(cb_eps).pop_front(1);
}
