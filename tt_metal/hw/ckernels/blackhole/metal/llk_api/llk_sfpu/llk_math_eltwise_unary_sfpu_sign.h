// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_sign.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

SFPU_UNARY_ONE_PARAM_KERNEL(sign, RC, uint exponent_size_8, exponent_size_8)
