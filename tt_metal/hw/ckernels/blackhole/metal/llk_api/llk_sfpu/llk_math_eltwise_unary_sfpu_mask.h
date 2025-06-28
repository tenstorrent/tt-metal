// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_mask.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

SFPU_UNARY_KERNEL(mask)

SFPU_DIM_SWITCH_KERNEL(mask, DataFormat, calculate_mask, Float16_b, calculate_int_mask, Int32)

SFPU_UNARY_KERNEL(mask_posinf)
