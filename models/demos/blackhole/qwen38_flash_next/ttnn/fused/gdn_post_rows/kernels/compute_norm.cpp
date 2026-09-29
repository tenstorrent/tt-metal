// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// ``post_norm`` compute: ``ttnn.rms_norm(o16 [1, 12, T, 128] bf16, weight=norm, epsilon=1e-6)`` followed by
// ``ttnn.multiply(normalized, sigmoid_bf16)``, call for call, on the four column tiles of one (value head, tile row).
//
// The norm is layernorm.cpp's RMSNORM + FUSE_GAMMA path at the op's DEFAULT compute config (rmsnorm.cpp:16-20:
// HiFi4, math_approx_mode = TRUE, fp32_dest_acc_en = FALSE), so this kernel runs in a 16-bit DEST with bf16
// intermediates and block_size 8 (layernorm_op_multi_core.cpp:239) -- Wt = 4 is one block.  The op reserves and
// pushes the block's full 8 slots for those 4 tiles (blocked_range's remainder bookkeeping); this kernel reserves
// 4, which changes no operand of the four mul_tiles or the four reduce_tile calls.  The kernel itself
// compiles APPROX = false (the gate's ``mul_binary_tile`` is binary_ng's, whose config is Precise), so the norm's
// two SFPU calls are written through the LLK macros with APPROX = true spelled out: the API wrappers
// ``mul_unary_tile`` / ``rsqrt_tile`` would hand the kernel's own APPROX to the ckernel functions.
//
// The gate is binary_ng's eltwise_binary_sfpu_no_bcast.cpp: copy both operands to DST, ``mul_binary_tile``
// (the fp32 product with a software RNE narrowing and the ``0 * x -> +0`` clamp), one pack.
//
// CBs: CB_X (0, bf16, the o16 tiles), CB_SIG (1), CB_SCALER (2), CB_EPS (3), CB_GAMMA (4), CB_XMM2 (6),
// CB_EX2 (7), CB_EX2PE (8), CB_FUSION (9), CB_NRM (10), CB_OUT (16); CB_PROJ (5) and CB_HIST (17) are the
// writer's history path and never reach the compute.  Runtime args: 0 units on this core.

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
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
#include "../../kernels/zones.h"

using namespace ckernel;

namespace {
constexpr uint32_t CB_X = 0, CB_SIG = 1, CB_SCALER = 2, CB_EPS = 3, CB_GAMMA = 4;
constexpr uint32_t CB_XMM2 = 6, CB_EX2 = 7, CB_EX2PE = 8, CB_FUSION = 9, CB_NRM = 10, CB_OUT = 16;
constexpr uint32_t HEAD_TILES = 4;  // Wt: 128 / 32
constexpr uint32_t dst0 = 0, dst1 = 1;
// numeric.h's row_wise_mean epilogue: scale_dest(dst0, bit_cast<uint32_t>(1.0f / W)) with W = 128
constexpr uint32_t RECIP_W = 0x3C000000u;
}  // namespace

void kernel_main() {
    const uint32_t units = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(CB_X, CB_X, CB_XMM2);
    cb_wait_front(CB_EPS, 1);

    for (uint32_t u = 0; u < units; ++u) {
        cb_wait_front(CB_X, HEAD_TILES);

        // x * x -> cb_xmm2 (bf16)
        {
            FUSED_ZONE("fz_gpo_cn_square");
            reconfig_data_format(CB_X, CB_X);
            pack_reconfig_data_format(CB_XMM2);
            mul_init(CB_X, CB_X);
            tile_regs_acquire();
            for (uint32_t i = 0; i < HEAD_TILES; ++i) {
                mul_tiles(CB_X, CB_X, i, i, i);
            }
            tile_regs_commit();
            cb_reserve_back(CB_XMM2, HEAD_TILES);
            tile_regs_wait();
            for (uint32_t i = 0; i < HEAD_TILES; ++i) {
                pack_tile(i, CB_XMM2);
            }
            tile_regs_release();
            cb_push_back(CB_XMM2, HEAD_TILES);
        }
        reconfig_data_format(CB_X, CB_XMM2, CB_X, CB_SCALER);

        // mean(x * x): numeric::row_wise_mean<SUM, REDUCE_ROW, FLOAT32_REDUCTION = false, FullBlockWithPopPolicy>
        {
            FUSED_ZONE("fz_gpo_cn_mean");
            cb_wait_front(CB_SCALER, 1);
            reconfig_data_format(CB_XMM2, CB_SCALER);
            tile_regs_acquire();
            reconfig_data_format(CB_SCALER, CB_XMM2);  // REDUCE_ROW SUM swaps the operands: scaler SrcA, data SrcB
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_XMM2, CB_SCALER, CB_EX2);
            cb_wait_front(CB_XMM2, HEAD_TILES);
            for (uint32_t j = 0; j < HEAD_TILES; ++j) {
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_XMM2, CB_SCALER, j, 0, dst0);
            }
            cb_pop_front(CB_XMM2, HEAD_TILES);
            // numeric.h's own order: accumulate_compute_loop ends with reduce_uninit + reconfig and the epilogue
            // (scale_dest) runs after it, still inside the same tile_regs_acquire
            reduce_uninit();
            reconfig_data_format(CB_XMM2, CB_SCALER);
            // numeric.h's scale_dest = binop_with_scalar_tile_init() + mul_unary_tile(dst0, 1/W).
            // binop_with_scalar.h:60-68 substitutes the KERNEL's APPROX into the template tuple of
            // SFPU_UNARY_CALL (llk_math_eltwise_unary_sfpu_macros.h:41-47), and there is no
            // llk_math_eltwise_unary_sfpu_binop_with_scalar free function outside tt_metal/hw/ckernels/quasar/,
            // so the op's own call with true written in place of APPROX IS the explicit form:
            // ckernel::sfpu::calculate_binop_with_scalar<true, MUL_UNARY, 8, false>.
            binop_with_scalar_tile_init();
            MATH(SFPU_UNARY_CALL(
                DST_SYNC_MODE,
                false /* is_fp32_dest_acc_en */,
                calculate_binop_with_scalar,
                (true /* APPROX */, MUL_UNARY, 8 /* ITERATIONS */, false /* is_fp32_dest_acc_en */),
                dst0,
                VectorMode::RC,
                RECIP_W));
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_EX2, 1);
            pack_reconfig_data_format(CB_EX2);
            pack_tile(dst0, CB_EX2);
            tile_regs_release();
            cb_push_back(CB_EX2, 1);
        }

        // rsqrt(mean + eps); the eps tile is the fp32 bits of 1e-6 truncated to bf16 (generate_bcast_col_scalar)
        {
            FUSED_ZONE("fz_gpo_cn_rsqrt");
            cb_wait_front(CB_EX2, 1);
            reconfig_data_format(CB_EX2, CB_EPS);
            tile_regs_acquire();
            add_init(CB_EX2, CB_EPS);
            add_tiles(CB_EX2, CB_EPS, 0, 0, dst0);
            // rsqrt_tile_init<false>() / rsqrt_tile<false>(dst0) as layernorm.cpp calls them, with the op's
            // APPROX = true written in place of the kernel's: rsqrt.h:18-20 and :37-43 pass APPROX into
            // sfpu::rsqrt_init and
            // sfpu::calculate_rsqrt through SFPU_UNARY_INIT_FN / SFPU_UNARY_CALL, so these two lines instantiate
            // ckernel::sfpu::rsqrt_init<true, false> and ckernel::sfpu::calculate_rsqrt<true, 8, false, false, false>
            // (ckernel_sfpu_rsqrt.h:19-32) -- the 10-bit table the op's math_approx_mode = TRUE selects.
            MATH(SFPU_UNARY_INIT_FN(rsqrt, sfpu::rsqrt_init, (true /* APPROX */, false /* legacy_compat */)));
            MATH(SFPU_UNARY_CALL(
                DST_SYNC_MODE,
                false /* is_fp32_dest_acc_en */,
                calculate_rsqrt,
                (true /* APPROX */,
                 8 /* ITERATIONS */,
                 false /* is_fp32_dest_acc_en */,
                 false /* FAST_APPROX */,
                 false /* legacy_compat */),
                dst0,
                VectorMode::RC));
            tile_regs_commit();
            cb_pop_front(CB_EX2, 1);
            cb_reserve_back(CB_EX2PE, 1);
            pack_reconfig_data_format(CB_EX2PE);
            tile_regs_wait();
            pack_tile(dst0, CB_EX2PE);
            tile_regs_release();
            cb_push_back(CB_EX2PE, 1);
        }

        // x * rsqrt (FPU column broadcast) -> cb_fusion, then * gamma (FPU row broadcast) -> cb_nrm
        {
            FUSED_ZONE("fz_gpo_cn_scale");
            cb_wait_front(CB_EX2PE, 1);
            reconfig_data_format(CB_X, CB_EX2PE);
            pack_reconfig_data_format(CB_FUSION);
            cb_reserve_back(CB_FUSION, HEAD_TILES);
            reconfig_data_format_srca(CB_FUSION, CB_X);
            tile_regs_acquire();
            mul_bcast_cols_init(CB_X, CB_EX2PE);
            for (uint32_t i = 0; i < HEAD_TILES; ++i) {
                mul_tiles_bcast_cols(CB_X, CB_EX2PE, i, 0, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < HEAD_TILES; ++i) {
                pack_tile(i, CB_FUSION);
            }
            tile_regs_release();
            cb_push_back(CB_FUSION, HEAD_TILES);
            reconfig_data_format_srca(CB_X, CB_FUSION);
        }

        {
            FUSED_ZONE("fz_gpo_cn_gamma");
            pack_reconfig_data_format(CB_NRM);
            reconfig_data_format_srcb(CB_EX2PE, CB_GAMMA);
            cb_wait_front(CB_GAMMA, HEAD_TILES);
            cb_wait_front(CB_FUSION, HEAD_TILES);
            tile_regs_acquire();
            mul_bcast_rows_init(CB_FUSION, CB_GAMMA);
            for (uint32_t i = 0; i < HEAD_TILES; ++i) {
                mul_tiles_bcast_rows(CB_FUSION, CB_GAMMA, i, i, i);
            }
            tile_regs_commit();
            cb_pop_front(CB_FUSION, HEAD_TILES);
            cb_reserve_back(CB_NRM, HEAD_TILES);
            tile_regs_wait();
            for (uint32_t i = 0; i < HEAD_TILES; ++i) {
                pack_tile(i, CB_NRM);
            }
            tile_regs_release();
            cb_push_back(CB_NRM, HEAD_TILES);
            cb_pop_front(CB_EX2PE, 1);
            cb_pop_front(CB_X, HEAD_TILES);
        }

        // the gate: ttnn.multiply(normalized, sigmoid_bf16) = binary_ng's SFPU multiply, one tile per acquire
        {
            FUSED_ZONE("fz_gpo_cn_gate");
            cb_wait_front(CB_NRM, HEAD_TILES);
            cb_wait_front(CB_SIG, HEAD_TILES);
            cb_reserve_back(CB_OUT, HEAD_TILES);
            pack_reconfig_data_format(CB_OUT);
            for (uint32_t i = 0; i < HEAD_TILES; ++i) {
                tile_regs_acquire();
                reconfig_data_format_srca(CB_SIG, CB_NRM);
                copy_init(CB_NRM);
                copy_tile(CB_NRM, i, dst0);
                reconfig_data_format_srca(CB_NRM, CB_SIG);
                copy_init(CB_SIG);
                copy_tile(CB_SIG, i, dst1);
                mul_binary_tile_init();
                mul_binary_tile(0, 1, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(dst0, CB_OUT);
                tile_regs_release();
            }
            cb_push_back(CB_OUT, HEAD_TILES);
            cb_pop_front(CB_NRM, HEAD_TILES);
            cb_pop_front(CB_SIG, HEAD_TILES);
        }
    }

    // the three constant tiles are read once and never popped inside the loop (the op's own gamma / scaler rule)
    cb_pop_front(CB_GAMMA, HEAD_TILES);
    cb_pop_front(CB_SCALER, 1);
    cb_pop_front(CB_EPS, 1);
}
