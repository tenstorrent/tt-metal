// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Option 4 — unary SFPU macros (per-tile + init).

#pragma once

#include "llk_math_eltwise_unary_sfpu_common_option4.h"
#include "llk_math_eltwise_unary_sfpu_init.h"

#define SFPU_UNARY_KERNEL_V2(IN_REG, OUT_REG, FN_PTR, DST_IDX, IS_32BIT_MODE, ...) \
    _llk_math_eltwise_unary_sfpu_params_<IN_REG, OUT_REG>(FN_PTR, DST_IDX, IS_32BIT_MODE, ##__VA_ARGS__)

#define SFPU_UNARY_KERNEL_INIT(OP) llk_math_eltwise_unary_sfpu_init<SfpuType::OP>()

#define SFPU_TEMPLATE_INIT_KERNEL(OP, INIT_CB, APPROX, SCALE, CLAMP_NEGATIVE) \
    llk_math_eltwise_unary_sfpu_init<SfpuType::OP>(INIT_CB<APPROX, SCALE, CLAMP_NEGATIVE>)
