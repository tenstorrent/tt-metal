// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_sfpu_types.h"
#include "llk_defs.h"
#include "llk_math_eltwise_unary_sfpu.h"

namespace ckernel {

template <SfpuType sfpu_op, ckernel::ApproximationMode APPROX_MODE>
inline void llk_math_eltwise_unary_sfpu_init() {
    _llk_math_eltwise_unary_sfpu_init_<sfpu_op>();
}

template <SfpuType sfpu_op, ckernel::ApproximationMode APPROX_MODE, class F, class... ARGS>
inline void llk_math_eltwise_unary_sfpu_init(F&& init_func, ARGS&&... args) {
    _llk_math_eltwise_unary_sfpu_init_<sfpu_op>();
    init_func(static_cast<ARGS&&>(args)...);
}
}  // namespace ckernel
