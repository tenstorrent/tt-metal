// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_polygamma.h"

namespace ckernel {

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_polygamma_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::polygamma, APPROXIMATE>(
        sfpu::polygamma_init<APPROXIMATE, is_fp32_dest_acc_en>);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_polygamma(
    uint32_t dst_index, uint32_t n_packed, uint32_t scale_packed, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_polygamma<APPROXIMATE, is_fp32_dest_acc_en>,
        dst_index,
        vector_mode,
        n_packed,
        scale_packed);
}

}  // namespace ckernel
