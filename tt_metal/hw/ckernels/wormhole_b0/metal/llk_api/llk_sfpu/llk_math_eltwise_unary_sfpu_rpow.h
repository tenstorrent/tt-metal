// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_rpow.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_rpow_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::rpow, APPROXIMATE>(sfpu::sfpu_binary_pow_init<APPROXIMATE>);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en = false>
inline void llk_math_eltwise_unary_sfpu_rpow(uint dst_index, uint32_t base_val, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        sfpu::calculate_rpow<APPROXIMATE, 8, is_fp32_dest_acc_en>, dst_index, vector_mode, base_val);
}

}  // namespace ckernel
