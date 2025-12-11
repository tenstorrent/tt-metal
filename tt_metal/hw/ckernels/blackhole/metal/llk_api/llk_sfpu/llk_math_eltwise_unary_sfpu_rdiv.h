// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_rdiv.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_rdiv_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::rdiv, APPROXIMATE>(sfpu::rdiv_init<APPROXIMATE>);
}

template <bool APPROXIMATE, bool fp32_dest_acc_en, RoundingMode rounding_mode, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_rdiv(
    uint32_t dst_index, uint32_t value, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_rdiv<APPROXIMATE, fp32_dest_acc_en, rounding_mode, ITERATIONS>,
        dst_index,
        vector_mode,
        value);
}

}  // namespace ckernel
