// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_dropout.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

namespace ckernel {

// New LLK SFPU APIs

SFPU_INIT_ONE_PARAM_KERNEL(dropout, sfpu::dropout_init, uint seed = 0, seed)

SFPU_UNARY_PARAMS_KERNEL_ONLY_COMPUTE(
    dropout, RC, uint integer_probability, uint scale_factor, integer_probability, scale_factor)

}  // namespace ckernel
