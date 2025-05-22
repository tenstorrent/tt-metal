// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_reshuffle_rows.h"

namespace ckernel {

// New LLK SFPU APIs

SFPU_INIT(reshuffle_rows)

SFPU_CALCULATE_RC_CUSTOM(reshuffle_rows, calculate_reshuffle_rows, PARAM_LIST(), PARAM_LIST(PARAM(uint32_t, idx_addr)))

}  // namespace ckernel
