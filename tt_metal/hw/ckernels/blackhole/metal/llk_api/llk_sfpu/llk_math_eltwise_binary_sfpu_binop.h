// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_binary.h"

namespace ckernel {

template <ckernel::ApproximationMode APPROX_MODE, ckernel::BinaryOp BINOP>
inline void llk_math_eltwise_binary_sfpu_binop_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROX_MODE>(
        ckernel::sfpu::sfpu_binary_init<APPROX_MODE, BINOP>);
}

template <ckernel::ApproximationMode APPROX_MODE, ckernel::BinaryOp BINOP, bool is_fp32_dest_acc_en = false>
inline void llk_math_eltwise_binary_sfpu_binop(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::calculate_sfpu_binary<APPROX_MODE, BINOP, 8, is_fp32_dest_acc_en>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}

template <ckernel::ApproximationMode APPROX_MODE, ckernel::BinaryOp BINOP, bool is_fp32_dest_acc_en = false>
inline void llk_math_eltwise_binary_sfpu_binop_mul(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::calculate_sfpu_binary_mul<APPROX_MODE, BINOP, 8, is_fp32_dest_acc_en>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}

template <ckernel::ApproximationMode APPROX_MODE, ckernel::BinaryOp BINOP, bool is_fp32_dest_acc_en = false>
inline void llk_math_eltwise_binary_sfpu_binop_div(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::calculate_sfpu_binary_div<APPROX_MODE, BINOP, 8, is_fp32_dest_acc_en>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}

}  // namespace ckernel
