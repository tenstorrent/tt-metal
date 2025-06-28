// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

SFPU_ROUNDING_KERNEL(floor, 8, false)
SFPU_ROUNDING_KERNEL(floor_float32, 8, true)
SFPU_ROUNDING_KERNEL(ceil, 8, false)
SFPU_ROUNDING_KERNEL(ceil_float32, 8, true)
SFPU_ROUNDING_KERNEL(trunc, 8, false)
SFPU_ROUNDING_KERNEL(trunc_float32, 8, true):
SFPU_UNARY_PARAMS_KERNEL(round, RC, int decimals, decimals)
