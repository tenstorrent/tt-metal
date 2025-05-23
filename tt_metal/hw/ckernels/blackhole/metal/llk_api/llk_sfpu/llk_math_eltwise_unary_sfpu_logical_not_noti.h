// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_logical_not_noti.h"

namespace ckernel {

// New LLK SFPU APIs

SFPU_INIT(logical_not_unary)

SFPU_CALCULATE_ALWAYS_RC(logical_not_unary_op, calculate_logical_not_unary, PARAM_LIST(), PARAM_LIST())

}  // namespace ckernel
