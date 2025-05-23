// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_isinf_isnan.h"

namespace ckernel {

// New LLK SFPU APIs

// isinf
SFPU_INIT(isinf)

SFPU_CALCULATE_ALWAYS_RC_APPROX_LAST(isinf, calculate_sfpu_isinf_isnan, PARAM_LIST(ARG(SfpuType::isinf)), PARAM_LIST())

// isposinf
SFPU_INIT(isposinf)

SFPU_CALCULATE_ALWAYS_RC_APPROX_LAST(isposinf, calculate_sfpu_isinf_isnan, PARAM_LIST(ARG(SfpuType::isposinf)), PARAM_LIST())

// isneginf
SFPU_INIT(isneginf)

SFPU_CALCULATE_ALWAYS_RC_APPROX_LAST(isneginf, calculate_sfpu_isinf_isnan, PARAM_LIST(ARG(SfpuType::isneginf)), PARAM_LIST())

// isnan
SFPU_INIT(isnan)

SFPU_CALCULATE_ALWAYS_RC_APPROX_LAST(isnan, calculate_sfpu_isinf_isnan, PARAM_LIST(ARG(SfpuType::isnan)), PARAM_LIST())

// isfinite
SFPU_INIT(isfinite)

SFPU_CALCULATE_ALWAYS_RC_APPROX_LAST(isfinite, calculate_sfpu_isinf_isnan, PARAM_LIST(ARG(SfpuType::isfinite)), PARAM_LIST())

}  // namespace ckernel
