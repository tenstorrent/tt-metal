// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "ckernel_sfpu_binary.h"
#include "llk_assert.h"

namespace ckernel {

template <bool APPROXIMATE, ckernel::BinaryOp BINOP>
inline void llk_math_eltwise_binary_sfpu_binop_init() {
    _llk_math_eltwise_sfpu_init_();
    ckernel::sfpu::sfpu_binary_init<APPROXIMATE, BINOP>();
}

template <bool APPROXIMATE, ckernel::BinaryOp BINOP, bool is_fp32_dest_acc_en = false>
inline void llk_math_eltwise_binary_sfpu_binop_mul(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
    LLK_ASSERT(vector_mode == VectorMode::RC, "Quasar currently only supports vector mode RC");
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_sfpu_binary<APPROXIMATE, BINOP, is_fp32_dest_acc_en>,
        0 /*dst_tile_index for addressing*/,
        SFPU_ITERATIONS,
        dst_index0,
        dst_index1,
        odst);
}

template <bool APPROXIMATE, ckernel::BinaryOp BINOP, bool is_fp32_dest_acc_en = false>
inline void llk_math_eltwise_binary_sfpu_binop_div(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
    LLK_ASSERT(vector_mode == VectorMode::RC, "Quasar currently only supports vector mode RC");
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_sfpu_binary<APPROXIMATE, BINOP, is_fp32_dest_acc_en>,
        0 /*dst_tile_index for addressing*/,
        SFPU_ITERATIONS,
        dst_index0,
        dst_index1,
        odst);
}

}  // namespace ckernel
