// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_dropout.h"

namespace ckernel {

// New LLK SFPU APIs

SFPU_INIT_WITH_FN(dropout, DEFAULT_PARAM(uint, seed, 0))

SFPU_CALCULATE(dropout, PARAM(uint, integer_probability), PARAM(uint, scale_factor))

}  // namespace ckernel
