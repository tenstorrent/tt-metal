// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_unary_max_min.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

SFPU_UNARY_PARAMS_KERNEL(unary_max, RC, uint param0, param0)
SFPU_UNARY_PARAMS_KERNEL(unary_min, RC, uint param0, param0)
SFPU_UNARY_INT32_KERNEL(unary_max)
