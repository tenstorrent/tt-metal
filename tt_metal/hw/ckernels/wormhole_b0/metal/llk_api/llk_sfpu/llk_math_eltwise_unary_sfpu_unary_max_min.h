// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_unary_max_min.h"

namespace ckernel {

// New LLK SFPU APIs

// Unary maximum
SFPU_INIT(unary_max)

SFPU_CALCULATE_RC_APPROX_LAST(unary_max, calculate_unary_max_min, PARAM_LIST(ARG(true)), PARAM_LIST(PARAM(uint, param0)))

// Unary minimum
SFPU_INIT(unary_min)

SFPU_CALCULATE_RC_APPROX_LAST(unary_min, calculate_unary_max_min, PARAM_LIST(ARG(false)), PARAM_LIST(PARAM(uint, param0)))

}  // namespace ckernel
