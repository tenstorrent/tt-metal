// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE, bool is_fp32_dest_acc_en = false /*unused*/>
inline void llk_math_eltwise_unary_sfpu_reciprocal(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_reciprocal<APPROXIMATE>,
        dst_index,
        vector_mode);

}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_reciprocal_init() {
    llk_math_eltwise_unary_sfpu_init<APPROXIMATE>();
}

}
