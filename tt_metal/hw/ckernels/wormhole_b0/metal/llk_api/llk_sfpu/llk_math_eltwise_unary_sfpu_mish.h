// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_mish.h"

namespace ckernel {

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_mish_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::mish>(sfpu::mish_init<APPROXIMATE, is_fp32_dest_acc_en>);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_mish(uint32_t dst_index, VectorMode vector_mode = VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::calculate_mish<APPROXIMATE, is_fp32_dest_acc_en>, dst_index, vector_mode);
}

}  // namespace ckernel
