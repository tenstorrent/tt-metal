// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_rsqrt.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE, bool legacy_compat>
inline void llk_math_eltwise_unary_sfpu_rsqrt_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::rsqrt, APPROXIMATE>(sfpu::rsqrt_init<APPROXIMATE, legacy_compat>);
}

template <bool APPROXIMATE, bool fp32_dest_acc_en, bool legacy_compat>
inline void llk_math_eltwise_unary_sfpu_rsqrt(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_rsqrt<APPROXIMATE, 8, fp32_dest_acc_en, legacy_compat>, dst_index, vector_mode);
}

}  // namespace ckernel
