// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// N-core compute: per tile row, ttnn.typecast(bf8 -> bf16) then ttnn.rms_norm (q_norm), the two
// ops _qwen_local_q_prep runs before RoPE.
//
// Runs with the q_norm compute config (RMSNorm.compute_kernel_config_hifi2, models/common/rmsnorm.py):
// HiFi2, fp32 DEST, math_approx_mode False. The RMSNorm part mirrors the stock interleaved kernel
// layernorm.cpp built with RMSNORM + FUSE_GAMMA, Wt = 4, block_size = 4, float32 intermediates
// (layernorm_op_multi_core.cpp: interm format Float32 when fp32_dest_acc_en, tile-reduction backend
// because LayerNormDefaultProgramConfig.use_welford is false). Each stage lists the stock line it
// mirrors; the call sequence, operand order, data formats and inits are the same, so the bits are too.
//
// Typecast stage: ttnn.typecast bf8 -> bf16 is a plain copy (typecast_tile<Bfp8_b, Float16_b> has no
// SFPU work, eltwise_unary/typecast.h). Every bf8 value is exactly representable in bf16, so the copy
// through fp32 DEST here gives the same bf16 tiles the stock typecast writes to DRAM.
//
// Compile-time args: 0 inv_w (fp32 bits of 1/128)
// Runtime args: 0 num_rows

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "qprep_common.hpp"

using namespace qprep;

// CB_X <- copy(CB_QIN): the bf8 -> bf16 typecast.
FORCE_INLINE void typecast_row() {
    reconfig_data_format_srca(CB_QIN);
    pack_reconfig_data_format(CB_X);
    copy_init(CB_QIN);
    cb_wait_front(CB_QIN, WT);
    tile_regs_acquire();
    for (uint32_t i = 0; i < WT; ++i) {
        copy_tile(CB_QIN, i, i);
    }
    tile_regs_commit();
    cb_pop_front(CB_QIN, WT);
    cb_reserve_back(CB_X, WT);
    tile_regs_wait();
    for (uint32_t i = 0; i < WT; ++i) {
        pack_tile(i, CB_X);
    }
    tile_regs_release();
    cb_push_back(CB_X, WT);
}

// CB_XMM2 <- CB_X * CB_X; CB_X stays. layernorm.cpp:207-208 (reconfig) and :243 (ckl::square:
// mul_init(x, x, acc_to_dest = false), Cumulative wait, no pop, one block of 4).
FORCE_INLINE void square_row() {
    reconfig_data_format(CB_X, CB_X);
    pack_reconfig_data_format(CB_XMM2);
    mul_init(CB_X, CB_X, false);
    cb_wait_front(CB_X, WT);
    tile_regs_acquire();
    for (uint32_t i = 0; i < WT; ++i) {
        mul_tiles(CB_X, CB_X, i, i, i);
    }
    tile_regs_commit();
    cb_reserve_back(CB_XMM2, WT);
    tile_regs_wait();
    for (uint32_t i = 0; i < WT; ++i) {
        pack_tile(i, CB_XMM2);
    }
    tile_regs_release();
    cb_push_back(CB_XMM2, WT);
}

// CB_EX2 <- sum_row(CB_XMM2) * 1/W; pops CB_XMM2. layernorm.cpp:256 (reconfig) and :261
// (numeric::row_wise_mean with FullBlockWithPopPolicy, kernel_util/compute/numeric.h: wait scaler,
// reconfig(in, scaler), acquire, swapped reconfig(scaler, in), reduce_init, reduce_tile x Wt, pop,
// reduce_uninit, reconfig(in, scaler), mul_unary_tile(1/N), pack, then wait_front on the output).
template <uint32_t inv_w>
FORCE_INLINE void mean_sq_row() {
    reconfig_data_format(CB_X, CB_XMM2, CB_X, CB_SCALER);
    cb_wait_front(CB_SCALER, 1);
    reconfig_data_format(CB_XMM2, CB_SCALER);
    tile_regs_acquire();
    reconfig_data_format(CB_SCALER, CB_XMM2);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_XMM2, CB_SCALER, CB_EX2);
    cb_wait_front(CB_XMM2, WT);
    for (uint32_t j = 0; j < WT; ++j) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_XMM2, CB_SCALER, j, 0, 0);
    }
    cb_pop_front(CB_XMM2, WT);
    reduce_uninit();
    reconfig_data_format(CB_XMM2, CB_SCALER);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_w);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(CB_EX2, 1);
    pack_reconfig_data_format(CB_EX2);
    pack_tile(0, CB_EX2);
    tile_regs_release();
    cb_push_back(CB_EX2, 1);
    cb_wait_front(CB_EX2, 1);
}

// CB_EX2PE <- rsqrt(CB_EX2 + eps); pops CB_EX2. layernorm.cpp:268-275 (eltwise_chain: BinaryFpu Add
// (add_init acc_to_dest = false), Rsqrt<Exact> = rsqrt_tile<RsqrtMode::Default>, PackTile).
FORCE_INLINE void rstd_row() {
    reconfig_data_format(CB_EX2, CB_EPS);
    pack_reconfig_data_format(CB_EX2PE);
    cb_wait_front(CB_EX2, 1);
    tile_regs_acquire();
    add_init(CB_EX2, CB_EPS, false);
    add_tiles(CB_EX2, CB_EPS, 0, 0, 0);
    rsqrt_tile_init();
    rsqrt_tile(0);  // RsqrtMode::Default (ckl Rsqrt<Approx::Exact>)
    tile_regs_commit();
    cb_reserve_back(CB_EX2PE, 1);
    tile_regs_wait();
    pack_tile(0, CB_EX2PE);
    tile_regs_release();
    cb_push_back(CB_EX2PE, 1);
    cb_pop_front(CB_EX2, 1);
}

// CB_NOUT <- (CB_X * bcast_col(CB_EX2PE)) -> fp32 CB_FUSION -> * bcast_row(CB_GAMMA).
// layernorm.cpp:283-345 with one block of 4 (FUSE_GAMMA, no beta): the x * rstd product is packed
// to the fp32 fusion buffer and unpacked again before the gamma multiply, as in the stock kernel.
FORCE_INLINE void normalize_row() {
    // x * rstd (BinaryFpu Mul, srcB col broadcast; reconfig enabled on both inputs and the pack).
    reconfig_data_format(CB_X, CB_EX2PE);
    pack_reconfig_data_format(CB_FUSION);
    mul_bcast_cols_init(CB_X, CB_EX2PE);
    cb_wait_front(CB_X, WT);
    cb_wait_front(CB_EX2PE, 1);
    tile_regs_acquire();
    for (uint32_t i = 0; i < WT; ++i) {
        mul_tiles_bcast_cols(CB_X, CB_EX2PE, i, 0, i);
    }
    tile_regs_commit();
    cb_reserve_back(CB_FUSION, WT);
    tile_regs_wait();
    for (uint32_t i = 0; i < WT; ++i) {
        pack_tile(i, CB_FUSION);
    }
    tile_regs_release();
    cb_push_back(CB_FUSION, WT);

    // * gamma (layernorm.cpp:310, 316, 321, then the row-broadcast Mul chain).
    reconfig_data_format_srca(CB_X, CB_FUSION);
    pack_reconfig_data_format(CB_NOUT);
    reconfig_data_format_srcb(CB_EX2PE, CB_GAMMA);
    mul_bcast_rows_init(CB_FUSION, CB_GAMMA);
    cb_wait_front(CB_FUSION, WT);
    cb_wait_front(CB_GAMMA, WT);
    tile_regs_acquire();
    for (uint32_t i = 0; i < WT; ++i) {
        mul_tiles_bcast_rows(CB_FUSION, CB_GAMMA, i, i, i);
    }
    tile_regs_commit();
    cb_reserve_back(CB_NOUT, WT);
    tile_regs_wait();
    for (uint32_t i = 0; i < WT; ++i) {
        pack_tile(i, CB_NOUT);
    }
    tile_regs_release();
    cb_push_back(CB_NOUT, WT);
    cb_pop_front(CB_FUSION, WT);

    // layernorm.cpp:387-388
    cb_pop_front(CB_EX2PE, 1);
    cb_pop_front(CB_X, WT);
}

void kernel_main() {
    constexpr uint32_t inv_w = get_compile_time_arg_val(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(0);

    // layernorm.cpp:155 (RMSNORM without pre-add): srcA = srcB = input buffer, pack = x^2 buffer.
    compute_kernel_hw_startup(CB_X, CB_X, CB_XMM2);
    cb_wait_front(CB_EPS, 1);

    for (uint32_t r = 0; r < num_rows; ++r) {
        typecast_row();
        square_row();
        mean_sq_row<inv_w>();
        rstd_row();
        normalize_row();
    }

    // The reader pushes the scaler, eps and gamma once (also when this core has no rows).
    cb_wait_front(CB_SCALER, 1);
    cb_wait_front(CB_GAMMA, WT);
    cb_pop_front(CB_SCALER, 1);
    cb_pop_front(CB_EPS, 1);
    cb_pop_front(CB_GAMMA, WT);
}
