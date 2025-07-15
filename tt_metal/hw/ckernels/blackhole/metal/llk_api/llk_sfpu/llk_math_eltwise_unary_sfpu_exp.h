// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_exp.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

SFPU_TEMPLATE_INIT_KERNEL(exponential, sfpu::exp_init)

SFPU_TEMPLATE_PARAMS_KERNEL(
    exponential,
    RC,
    bool FAST_APPROX,
    bool SCALE_EN,
    bool SKIP_POSITIVE_CHECK,
    int ITERATIONS,
    FAST_APPROX,
    SCALE_EN,
    SKIP_POSITIVE_CHECK,
    ITERATIONS)
