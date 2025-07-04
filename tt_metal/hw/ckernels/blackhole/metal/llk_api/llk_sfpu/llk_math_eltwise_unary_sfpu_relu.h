// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_relu.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

SFPU_UNARY_OP_INIT(relu, relu_min)
SFPU_UNARY_OP_INIT(relu_min, relu_min)
SFPU_UNARY_OP_INIT(relu_max, relu_max)
SFPU_UNARY_OP_INIT(lrelu, lrelu)

SFPU_UNARY_OP_COMPUTE(lrelu, calculate_lrelu)
SFPU_UNARY_OP_COMPUTE(relu_min, relu_min)
SFPU_UNARY_OP_COMPUTE(relu_max, relu_max)
SFPU_RELU_ALIAS()
