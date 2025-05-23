// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_i1.h"

namespace ckernel {

// New LLK SFPU APIs

SFPU_INIT(i1)

SFPU_CALCULATE_ALWAYS_RC(i1_op, calculate_i1, PARAM_LIST(), PARAM_LIST())

}  // namespace ckernel
