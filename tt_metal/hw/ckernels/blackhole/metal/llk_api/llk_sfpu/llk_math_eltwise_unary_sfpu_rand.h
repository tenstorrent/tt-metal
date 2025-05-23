// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_instr_params.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_rand.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace ckernel {

// New LLK SFPU APIs

SFPU_INIT_CUSTOM_NAME_WITH_FN(rand, unused, rand_init, DEFAULT_PARAM(uint32_t, seed, 0))

SFPU_CALCULATE_ALWAYS_RC(rand, rand, PARAM_LIST(), PARAM_LIST(PARAM(uint32_t, from), PARAM(uint32_t, scale)))

}  // namespace ckernel
