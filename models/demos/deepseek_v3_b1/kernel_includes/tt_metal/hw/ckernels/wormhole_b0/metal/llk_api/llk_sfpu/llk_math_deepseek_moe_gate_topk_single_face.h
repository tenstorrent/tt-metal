// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_deepseek_moe_gate_topk_single_face.h"

namespace ckernel {

template <ckernel::ApproximationMode APPROX_MODE, bool is_fp32_dest_acc_en>
inline void llk_math_sfpu_deepseek_moe_gate_topk_init() {
    // Don't need the second addrmod so set type to unused
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROX_MODE>(
        sfpu::deepseek_moe_gate_topk_init<APPROX_MODE, is_fp32_dest_acc_en>);
}

template <ckernel::ApproximationMode APPROX_MODE, bool is_fp32_dest_acc_en>
inline void llk_math_sfpu_deepseek_moe_gate_sum_top2(uint dst_index, int vector_mode = (int)VectorMode::RC_custom) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::deepseek_moe_gate_sum_top2<APPROX_MODE, is_fp32_dest_acc_en>, dst_index, vector_mode);
}

template <ckernel::ApproximationMode APPROX_MODE, bool is_fp32_dest_acc_en>
inline void llk_math_sfpu_deepseek_moe_gate_sort_top4_groups(
    uint dst_index, int vector_mode = (int)VectorMode::RC_custom) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::deepseek_moe_gate_sort_top4_groups<APPROX_MODE, is_fp32_dest_acc_en>, dst_index, vector_mode);
}

template <ckernel::ApproximationMode APPROX_MODE, bool is_fp32_dest_acc_en>
inline void llk_math_sfpu_deepseek_moe_gate_top8(
    uint dst_index, uint32_t eps, uint32_t scale, int vector_mode = (int)VectorMode::RC_custom) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::deepseek_moe_gate_top8<APPROX_MODE, is_fp32_dest_acc_en>, dst_index, vector_mode, eps, scale);
}

}  // namespace ckernel
