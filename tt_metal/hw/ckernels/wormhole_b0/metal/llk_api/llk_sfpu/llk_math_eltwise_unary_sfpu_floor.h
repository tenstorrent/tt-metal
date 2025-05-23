// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_floor.h"

namespace ckernel {

// New LLK SFPU APIs

SFPU_INIT(floor)

SFPU_CALCULATE(floor)

SFPU_CALCULATE(floor_float32)
}  // namespace ckernel
