// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_unary_comp.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

// int32 variants
SFPU_COMP_INT32_KERNEL(ne, unary_ne)
SFPU_COMP_INT32_KERNEL(eq, unary_eq)
SFPU_COMP_INT32_KERNEL_UNDERSCORE(gt, unary_gt)
SFPU_COMP_INT32_KERNEL_UNDERSCORE(lt, unary_lt)

// normal variants
SFPU_COMP_KERNEL(ne)
SFPU_COMP_KERNEL(eq)
SFPU_COMP_KERNEL(gt)
SFPU_COMP_KERNEL(lt)
