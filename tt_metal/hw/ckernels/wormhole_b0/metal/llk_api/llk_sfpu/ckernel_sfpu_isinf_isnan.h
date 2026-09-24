// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/ckernel_sfpu_isinf_isnan.h"

namespace ckernel::sfpu {

// Op class for the floating-point class tests (isinf, isposinf, isneginf, isnan, isfinite): 1.0 where the
// test holds, else 0.0.
template <bool APPROXIMATION_MODE, FiniteCheck CHECK, int ITERATIONS = 8>
struct IsinfIsnan : SfpuUnaryOp<IsinfIsnan<APPROXIMATION_MODE, CHECK, ITERATIONS>> {
    static inline __attribute__((always_inline)) void calculate() {
        _calculate_sfpu_isinf_isnan_<CHECK, APPROXIMATION_MODE, ITERATIONS>();
    }
};

}  // namespace ckernel::sfpu
