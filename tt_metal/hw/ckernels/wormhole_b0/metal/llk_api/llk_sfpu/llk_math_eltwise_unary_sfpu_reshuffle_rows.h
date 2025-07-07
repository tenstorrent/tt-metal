// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_reshuffle_rows.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_reshuffle_rows_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::reshuffle_rows, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_reshuffle_rows(
    uint dst_index, uint32_t idx_addr, int vector_mode = (int)VectorMode::RC_custom) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_reshuffle_rows<APPROXIMATE>, dst_index, vector_mode, idx_addr);
}

}  // namespace ckernel
