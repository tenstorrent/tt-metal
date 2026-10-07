// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared pieces of the two rmsnorm_bw compute kernels. Per-element math runs on the SFPU in fp32 DEST
// (4 fp32 tiles per acquire); the only FPU ops are the gamma row-broadcast multiply and 32x32 matmuls with
// constant tiles that turn a tile into a per-row value broadcast along its columns.

#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"
#include "tt-train/sources/ttml/metal/ops/rmsnorm_bw/device/kernels/rmsnorm_bw_cbs.hpp"

namespace cb = rmsnorm_bw_cb;

constexpr uint32_t work_count = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr uint32_t S = get_compile_time_arg_val(2);
constexpr uint32_t St = get_compile_time_arg_val(3);
constexpr uint32_t block = get_compile_time_arg_val(4);
constexpr uint32_t mask_w = get_compile_time_arg_val(5);
constexpr uint32_t inv_c_bits = get_compile_time_arg_val(6);  // fp32 bits of 1/C (logical C)

// Unpack one tile of `cb_id` (bf16 or fp32) into fp32 DEST register `reg`.
FORCE_INLINE void load_tile(uint32_t cb_id, uint32_t idx, uint32_t reg) {
    reconfig_data_format_srca(cb_id);
    copy_init(cb_id);
    copy_tile(cb_id, idx, reg);
}

// reg = gamma (row 0 of gamma tile `idx`, broadcast down the rows) * dy tile `idx`.
// mul_tiles_bcast_rows accumulates into DEST, so the register is zeroed first.
FORCE_INLINE void gamma_times_dy(uint32_t idx, uint32_t reg) {
    load_tile(cb::zero, 0, reg);
    reconfig_data_format(cb::dy, cb::gamma);
    mul_bcast_rows_init(cb::dy, cb::gamma);
    mul_tiles_bcast_rows(cb::dy, cb::gamma, idx, idx, reg);
}

// reg = X @ K for single tiles X (cb_x) and K (cb_k); K is a constant bf16 tile.
FORCE_INLINE void matmul_with_constant(uint32_t cb_x, uint32_t cb_k, uint32_t reg) {
    // matmul maps in0 -> SrcB, in1 -> SrcA, so reconfigure in (in1, in0) order.
    reconfig_data_format(cb_k, cb_x);
    matmul_init(cb_x, cb_k, 0);
    matmul_tiles(cb_x, cb_k, 0, 0, reg);
}

struct SliceGeometry {
    uint32_t col0;
    uint32_t ncols;
};

FORCE_INLINE SliceGeometry slice_for(uint32_t work) {
    const uint32_t r = work / S;
    const uint32_t s = work - r * S;
    const uint32_t col0 = s * St;
    return SliceGeometry{col0, (col0 + St <= Wt) ? St : (Wt - col0)};
}
