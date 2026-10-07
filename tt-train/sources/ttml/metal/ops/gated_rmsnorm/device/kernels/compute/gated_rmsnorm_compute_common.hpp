// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared pieces of the gated_rmsnorm forward/backward compute kernels.
//
// All per-element math runs on the SFPU in fp32 DEST (DEST holds 4 fp32 tiles per acquire in half-sync
// mode, so every acquire below uses registers 0..3 only). The only FPU op is a single 32x32 matmul with an
// all-ones tile, which turns a tile of partial sums into its row-sum broadcast to every column.

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"
#include "tt-train/sources/ttml/metal/ops/gated_rmsnorm/device/kernels/gated_rmsnorm_cbs.hpp"

namespace cb = gated_rmsnorm_cb;

constexpr uint32_t work_count = get_compile_time_arg_val(0);
constexpr uint32_t Gt = get_compile_time_arg_val(1);
constexpr uint32_t inv_group_bits = get_compile_time_arg_val(2);  // fp32 bits of 1/group
constexpr uint32_t eps_bits = get_compile_time_arg_val(3);        // fp32 bits of epsilon

// Unpack one tile of `cb_id` (bf16 or fp32) into fp32 DEST register `reg`.
FORCE_INLINE void load_tile(uint32_t cb_id, uint32_t idx, uint32_t reg) {
    reconfig_data_format_srca(cb_id);
    copy_init(cb_id);
    copy_tile(cb_id, idx, reg);
}

// Produces cb::inv = rsqrt(mean_c(x^2) + eps), broadcast to every column, for the group currently at the
// front of cb::x (Gt tiles). Leaves one tile in cb::inv; the caller pops it.
FORCE_INLINE void compute_inv_rms() {
    // sq = sum_j x_j * x_j  (elementwise across the Gt tiles of the group)
    tile_regs_acquire();
    mul_binary_tile_init();
    add_binary_tile_init();
    for (uint32_t j = 0; j < Gt; ++j) {
        const uint32_t reg = (j == 0) ? 0U : 1U;
        load_tile(cb::x, j, reg);
        mul_binary_tile(reg, reg, reg);
        if (j != 0) {
            add_binary_tile(0, 1, 0);
        }
    }
    tile_regs_commit();
    pack_and_push(0, cb::sq);

    // inv = rsqrt(rowsum(sq) / group + eps); rowsum via sq @ ones (result broadcast along columns).
    cb_wait_front(cb::sq, 1);
    tile_regs_acquire();
    // matmul maps in0 -> SrcB, in1 -> SrcA, so reconfigure in (in1, in0) order.
    reconfig_data_format(cb::ones, cb::sq);
    matmul_init(cb::sq, cb::ones, 0);
    matmul_tiles(cb::sq, cb::ones, 0, 0, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_group_bits);
    add_unary_tile(0, eps_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    pack_and_push(0, cb::inv);
    cb_pop_front(cb::sq, 1);
    cb_wait_front(cb::inv, 1);
}
