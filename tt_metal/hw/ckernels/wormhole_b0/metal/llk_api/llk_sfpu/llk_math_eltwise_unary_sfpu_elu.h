// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_elu.h"

namespace ckernel {

// New LLK SFPU APIs

SFPU_INIT_WITH_FN(elu)

SFPU_CALCULATE_ALWAYS_RC(elu, calculate_elu, PARAM_LIST(), PARAM_LIST(PARAM(uint, param0)))

}  // namespace ckernel
