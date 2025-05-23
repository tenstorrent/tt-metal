// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_relu.h"

namespace ckernel {

// New LLK SFPU APIs

SFPU_INIT_CUSTOM_NAME(relu, relu_min)

SFPU_INIT(lrelu)
SFPU_INIT(relu_max)

SFPU_INIT(relu_min)

SFPU_CALCULATE_ALWAYS_RC(lrelu, calculate_lrelu, PARAM_LIST(), PARAM_LIST(DEFAULT_PARAM(uint, param0, 0)))

SFPU_CALCULATE_ALWAYS_RC(relu_max, relu_max, PARAM_LIST(), PARAM_LIST(DEFAULT_PARAM(uint, param0, 0)))

SFPU_CALCULATE_ALWAYS_RC(relu_min, relu_min, PARAM_LIST(), PARAM_LIST(DEFAULT_PARAM(uint, param0, 0)))

SFPU_CALCULATE_ALWAYS_RC(relu, relu_min, PARAM_LIST(), PARAM_LIST(ARG(0)))

}  // namespace ckernel
