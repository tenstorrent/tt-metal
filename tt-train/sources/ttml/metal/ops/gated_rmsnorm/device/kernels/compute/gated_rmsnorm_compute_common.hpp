// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
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

namespace cb = ttml_gated_rmsnorm_cb;

constexpr uint32_t work_count = get_compile_time_arg_val(0);
constexpr uint32_t Gt = get_compile_time_arg_val(1);
constexpr uint32_t inv_group_bits = get_compile_time_arg_val(2);  // fp32 bits of 1/V
constexpr uint32_t eps_bits = get_compile_time_arg_val(3);        // fp32 bits of eps

// All math runs on fp32 DEST, which holds 4 tiles per acquire, so only registers 0..3 are used.
// CBs mix bf16 and fp32, so SrcA is reconfigured on every load.
inline void load_tile(const uint32_t cb_id, const uint32_t tile_idx, const uint32_t reg) {
    reconfig_data_format_srca(cb_id);
    copy_init(cb_id);
    copy_tile(cb_id, tile_idx, reg);
}

inline void mul_regs(const uint32_t a, const uint32_t b, const uint32_t out) {
    mul_binary_tile_init();
    mul_binary_tile(a, b, out);
}

inline void add_regs(const uint32_t a, const uint32_t b, const uint32_t out) {
    add_binary_tile_init();
    add_binary_tile(a, b, out);
}

inline void sub_regs(const uint32_t a, const uint32_t b, const uint32_t out) {
    sub_binary_tile_init();
    sub_binary_tile(a, b, out);
}

// DEST[reg] = (tile 0 of cb_in) @ ones: every column holds the row-sum. Matmul maps in0 to SrcB and
// in1 to SrcA, hence the reversed reconfig.
inline void row_sum_to_reg(const uint32_t cb_in, const uint32_t reg) {
    reconfig_data_format(cb::ones, cb_in);
    matmul_init(cb_in, cb::ones);
    matmul_tiles(cb_in, cb::ones, 0U, 0U, reg);
}

// cb::inv = rsqrt(sum_c(x_c^2) / V + eps) for the group at the front of cb::x. Leaves cb::inv waited.
inline void compute_inv_rms() {
    tile_regs_acquire();
    for (uint32_t j = 0; j < Gt; ++j) {
        const uint32_t reg = j == 0 ? 0U : 1U;
        load_tile(cb::x, j, reg);
        mul_regs(reg, reg, reg);
        if (j > 0) {
            add_regs(0U, 1U, 0U);
        }
    }
    tile_regs_commit();
    pack_and_push(0U, cb::sq);

    cb_wait_front(cb::sq, onetile);
    tile_regs_acquire();
    row_sum_to_reg(cb::sq, 0U);
    binop_with_scalar_tile_init();
    mul_unary_tile(0U, inv_group_bits);
    add_unary_tile(0U, eps_bits);
    rsqrt_tile_init();
    rsqrt_tile(0U);
    tile_regs_commit();
    cb_pop_front(cb::sq, onetile);
    pack_and_push(0U, cb::inv);
    cb_wait_front(cb::inv, onetile);
}
