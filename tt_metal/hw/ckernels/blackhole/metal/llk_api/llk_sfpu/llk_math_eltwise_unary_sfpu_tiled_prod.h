// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_tiled_prod.h"

namespace ckernel {

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_tiled_prod_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::tiled_prod, APPROX_MODE>();
}

template <ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_tiled_prod(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROX_MODE>(
        ckernel::sfpu::calculate_tiled_prod<APPROX_MODE>, dst_index, vector_mode);
}

}  // namespace ckernel
