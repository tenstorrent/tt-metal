// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_rsqrt.h"

namespace ckernel {

// New LLK SFPU APIs

SFPU_INIT_WITH_FN(rsqrt)

SFPU_CALCULATE_RC(rsqrt, calculate_rsqrt, PARAM_LIST(ARG(8), ARG(25)), PARAM_LIST())

}  // namespace ckernel
