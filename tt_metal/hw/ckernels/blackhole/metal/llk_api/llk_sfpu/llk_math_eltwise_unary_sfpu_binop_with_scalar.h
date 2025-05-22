// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_binop_with_unary.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_init.h"

namespace ckernel {

// New LLK SFPU APIs

SFPU_CALCULATE_RC(binop_with_scalar, calculate_binop_with_scalar, PARAM_LIST(PARAM(int, binop_mode), ARG(8)), PARAM_LIST(PARAM(uint32_t, param1)))

SFPU_INIT_CUSTOM_NAME(binop_with_scalar, unused)

}  // namespace ckernel
