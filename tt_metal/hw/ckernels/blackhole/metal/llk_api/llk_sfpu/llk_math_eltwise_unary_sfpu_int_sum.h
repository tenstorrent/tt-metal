// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_int_sum.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

enum SumIntDim { SUM_COL = 0, SUM_ROW };

SFPU_INIT_KERNEL_NOARG(sum_int, sfpu::sum_int_init)

SFPU_DIM_SWITCH_KERNEL(sum_int, SumIntDim, calculate_sum_int_col, R, calculate_sum_int_row, C)

SFPU_ONE_PARAM_CONST_ITERS_KERNEL(add_int, 8)
