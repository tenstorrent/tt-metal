// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_cumsum.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE /*unused*/>
inline void llk_math_eltwise_unary_sfpu_cumsum_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::cumsum, false>(
        sfpu::cumsum_init<false>);  // There is only non APPROXIMATE implementation
}

template <bool APPROXIMATE /*unused*/>
inline void llk_math_eltwise_unary_sfpu_cumsum(
    uint dst_index, bool first, int vector_mode = (int)VectorMode::RC_custom /*unused*/) {
    _llk_math_eltwise_unary_sfpu_params_<false>(
        ckernel::sfpu::calculate_cumsum<false>,  // There is only non APPROXIMATE implementation
        dst_index,
        VectorMode::RC_custom,  // Can only work in RC_custom mode
        first);
}

}  // namespace ckernel
