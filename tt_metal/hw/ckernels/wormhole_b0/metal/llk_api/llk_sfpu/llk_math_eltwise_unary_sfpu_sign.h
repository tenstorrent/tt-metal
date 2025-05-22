// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_sign.h"

namespace ckernel {

// New LLK SFPU APIs

SFPU_INIT(sign)

SFPU_CALCULATE_MANUAL(sign, calculate_sign, PARAM_LIST(),PARAM_LIST(), PARAM_LIST(DEFAULT_PARAM(int, vector_mode, (int)VectorMode::RC), DEFAULT_PARAM(uint, exponent_size_8, 1)))

}  // namespace ckernel
