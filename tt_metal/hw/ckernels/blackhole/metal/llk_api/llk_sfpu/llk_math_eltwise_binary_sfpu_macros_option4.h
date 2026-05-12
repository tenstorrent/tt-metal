// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Option 4 — binary SFPU macros (per-tile + init).

#pragma once

#include "llk_math_eltwise_binary_sfpu_common_option4.h"
#include "llk_math_eltwise_binary_sfpu_init.h"
#include "ckernel_sfpu_binary.h"

#define SFPU_BINARY_KERNEL_V2(IN0_REG, IN1_REG, OUT_REG, FN_PTR, DST_IN0, DST_IN1, DST_OUT, IS_32BIT_MODE, ...) \
    _llk_math_eltwise_binary_sfpu_params_<IN0_REG, IN1_REG, OUT_REG>(                                           \
        FN_PTR, DST_IN0, DST_IN1, DST_OUT, IS_32BIT_MODE, ##__VA_ARGS__)

#define SFPU_BINARY_KERNEL_INIT(BINOP)                   \
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused>( \
        ckernel::sfpu::sfpu_binary_init<APPROX, ckernel::BinaryOp::BINOP>)
