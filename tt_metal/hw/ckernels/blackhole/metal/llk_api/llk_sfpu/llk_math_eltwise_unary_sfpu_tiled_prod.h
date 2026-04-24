// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_tiled_prod.h"

namespace ckernel {

inline void llk_math_eltwise_unary_sfpu_tiled_prod_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::tiled_prod>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_tiled_prod(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::calculate_tiled_prod<APPROXIMATE>, dst_index, dst_index, vector_mode);
}

}  // namespace ckernel
