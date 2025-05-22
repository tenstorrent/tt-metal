// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_round.h"

namespace ckernel {

// New LLK SFPU APIs

SFPU_INIT(round)

SFPU_CALCULATE(round, PARAM(int, decimals))

}  // namespace ckernel
