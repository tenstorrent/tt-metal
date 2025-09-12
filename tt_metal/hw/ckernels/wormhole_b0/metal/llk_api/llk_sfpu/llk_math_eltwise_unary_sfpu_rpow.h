// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_rpow.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_rpow_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::rpow, APPROXIMATE>(
        // ckernel::sfpu::_init_exponential_<APPROXIMATE, /*FAST_APPROX=*/APPROXIMATE,
        // /*SCALE=*/p_sfpu::kCONST_1_FP16B>);
        ckernel::sfpu::sfpu_binary_init<APPROXIMATE, BinaryOp::POW>);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en = false>
inline void llk_math_eltwise_unary_sfpu_rpow(uint dst_index, uint32_t log_val, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_rpow<APPROXIMATE, 8, is_fp32_dest_acc_en>, dst_index, vector_mode, log_val);
}

}  // namespace ckernel
