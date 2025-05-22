// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_bitwise_xor.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_init.h"

namespace ckernel {

// New LLK SFPU APIs

SFPU_INIT(bitwise_xor)

SFPU_CALCULATE(bitwise_xor, PARAM(uint, param0))

}  // namespace ckernel
