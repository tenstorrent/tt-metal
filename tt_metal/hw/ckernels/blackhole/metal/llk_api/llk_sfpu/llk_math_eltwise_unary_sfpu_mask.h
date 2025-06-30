// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_mask.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

namespace ckernel {

// New LLK SFPU APIs

SFPU_UNARY_KERNEL_INIT_ONLY(mask)

SFPU_DIM_DUALTYPE_SWITCH_KERNEL(mask, DataFormat, calculate_mask, Float16_b, calculate_int_mask, Int32)

SFPU_UNARY_KERNEL_NO_INIT(mask_posinf)

}  // namespace ckernel
