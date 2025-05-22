// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_unary_comp.h"

namespace ckernel {

// New LLK SFPU APIs

// Unary Not equal
SFPU_CALCULATE_RC(unary_ne_int32, calculate_comp_unary_int, PARAM_LIST(ARG(SfpuType::unary_ne)), PARAM_LIST(PARAM(uint, param0)))

SFPU_INIT(unary_ne)

SFPU_CALCULATE(unary_ne, PARAM(uint, param0))

// Unary equal
SFPU_CALCULATE_RC(unary_eq_int32, calculate_comp_unary_int, PARAM_LIST(ARG(SfpuType::unary_eq)), PARAM_LIST(PARAM(uint, param0)))

SFPU_INIT(unary_eq)

SFPU_CALCULATE(unary_eq, PARAM(uint, param0))

// Unary greater than
SFPU_INIT(unary_gt)

SFPU_CALCULATE(unary_gt, PARAM(uint, param0))

// Unary lesser than
SFPU_INIT(unary_lt)

SFPU_CALCULATE(unary_lt, PARAM(uint, param0))
}  // namespace ckernel
