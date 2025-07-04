// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_fill.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

namespace ckernel {

// New LLK SFPU APIs

SFPU_UNARY_ONE_PARAM_KERNEL(fill, RC, float param0, param0)

SFPU_UNARY_PARAMS_KERNEL_WITH_CUSTOM_CALC(
    fill_bitcast, RC, ckernel::sfpu::calculate_fill_bitcast, uint32_t param0, param0)

}  // namespace ckernel
