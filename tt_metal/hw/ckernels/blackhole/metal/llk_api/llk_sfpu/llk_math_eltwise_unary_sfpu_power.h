// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_power_iterative.h"

namespace ckernel {

// New LLK SFPU APIs

SFPU_INIT(power)

SFPU_CALCULATE_RC(power, calculate_power_iterative, PARAM_LIST(), PARAM_LIST(DEFAULT_PARAM(int, pow, 0)))

}  // namespace ckernel
