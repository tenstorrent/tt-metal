// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_right_shift.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_init.h"

namespace ckernel {

// New LLK SFPU APIs

SFPU_INIT(right_shift)

SFPU_CALCULATE(right_shift, PARAM(uint, param0))

}  // namespace ckernel
