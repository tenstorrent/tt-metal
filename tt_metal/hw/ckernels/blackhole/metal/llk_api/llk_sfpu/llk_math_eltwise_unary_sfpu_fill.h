// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_fill.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_init.h"

namespace ckernel {

// New LLK SFPU APIs

SFPU_INIT(fill)

SFPU_CALCULATE(fill, PARAM(float, param0))

SFPU_CALCULATE(fill_bitcast, PARAM(uint32_t, param0))

}  // namespace ckernel
