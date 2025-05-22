// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_remainder.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_init.h"

namespace ckernel {

// New LLK SFPU APIs

SFPU_INIT_CUSTOM_NAME_WITH_FN(remainder, remainder, init_remainder, PARAM(uint, param0), PARAM(uint, param1))

SFPU_CALCULATE(remainder, PARAM(uint, param0), PARAM(uint, param1))

}  // namespace ckernel
