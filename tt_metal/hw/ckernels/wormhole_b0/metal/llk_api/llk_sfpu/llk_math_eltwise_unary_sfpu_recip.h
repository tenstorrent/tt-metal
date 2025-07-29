// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_reciprocal(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_reciprocal<APPROXIMATE, is_fp32_dest_acc_en, 8>, dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_reciprocal_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::reciprocal, APPROXIMATE>(sfpu::recip_init<APPROXIMATE>);
}

}  // namespace ckernel
