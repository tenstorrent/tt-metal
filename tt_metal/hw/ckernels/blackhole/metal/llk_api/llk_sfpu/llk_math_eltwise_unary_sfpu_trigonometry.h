// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_trigonometry.h"

namespace ckernel {

// New LLK SFPU APIs

// sine
SFPU_INIT(sine)

SFPU_CALCULATE_ALWAYS_RC_APPROX_LAST(sine_op, calculate_sfpu_trig, PARAM_LIST(ARG(SfpuType::sine)), PARAM_LIST())

// cosine
SFPU_INIT(cosine)

SFPU_CALCULATE_ALWAYS_RC_APPROX_LAST(cosine_op, calculate_sfpu_trig, PARAM_LIST(ARG(SfpuType::cosine)), PARAM_LIST())

// tangent
SFPU_INIT(tan)

SFPU_CALCULATE_ALWAYS_RC_APPROX_LAST(tan_op, calculate_sfpu_trig, PARAM_LIST(ARG(SfpuType::tan)), PARAM_LIST())

// asin
SFPU_INIT(asin)

SFPU_CALCULATE(asin)

// acos
SFPU_INIT(acos)

SFPU_CALCULATE(acos)

// atan
SFPU_INIT_WITH_FN(atan)

SFPU_CALCULATE(atan)

}  // namespace ckernel
