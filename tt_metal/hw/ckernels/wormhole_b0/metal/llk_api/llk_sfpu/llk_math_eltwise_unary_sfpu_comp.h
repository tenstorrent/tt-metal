// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_comp.h"

namespace ckernel {

// New LLK SFPU APIs

// EQZ
SFPU_CALCULATE_RC(eqz, calculate_comp, PARAM_LIST(ARG(SfpuType::equal_zero)), PARAM_LIST(ARG(8)))

SFPU_CALCULATE_RC(eqz_int32, calculate_comp_int, PARAM_LIST(ARG(SfpuType::equal_zero)), PARAM_LIST())

SFPU_INIT_CUSTOM_NAME(eqz, equal_zero)

// NEZ
SFPU_CALCULATE_RC(nez, calculate_comp, PARAM_LIST(ARG(SfpuType::not_equal_zero)), PARAM_LIST(ARG(8)))

SFPU_CALCULATE_RC(nez_int32, calculate_comp_int, PARAM_LIST(ARG(SfpuType::not_equal_zero)), PARAM_LIST())

SFPU_INIT_CUSTOM_NAME(nez, not_equal_zero)

// LTZ
SFPU_CALCULATE_RC(ltz, calculate_comp, PARAM_LIST(ARG(SfpuType::less_than_zero)), PARAM_LIST(ARG(8)))

SFPU_CALCULATE_RC(ltz_int32, calculate_comp_int, PARAM_LIST(ARG(SfpuType::less_than_zero)), PARAM_LIST())

SFPU_INIT_CUSTOM_NAME(ltz, less_than_zero)

// GTZ
SFPU_CALCULATE_RC(gtz, calculate_comp, PARAM_LIST(ARG(SfpuType::greater_than_zero)), PARAM_LIST(ARG(8)))

SFPU_CALCULATE_RC(gtz_int32, calculate_comp_int, PARAM_LIST(ARG(SfpuType::greater_than_zero)), PARAM_LIST())

SFPU_INIT_CUSTOM_NAME(gtz, greater_than_zero)

// LEZ
SFPU_CALCULATE_RC(lez, calculate_comp, PARAM_LIST(ARG(SfpuType::less_than_equal_zero)), PARAM_LIST(ARG(8)))

SFPU_CALCULATE_RC(lez_int32, calculate_comp_int, PARAM_LIST(ARG(SfpuType::less_than_equal_zero)), PARAM_LIST())

SFPU_INIT_CUSTOM_NAME(lez, less_than_equal_zero)

// GEZ
SFPU_CALCULATE_RC(gez, calculate_comp, PARAM_LIST(ARG(SfpuType::greater_than_equal_zero)), PARAM_LIST(ARG(8)))

SFPU_CALCULATE_RC(gez_int32, calculate_comp_int, PARAM_LIST(ARG(SfpuType::greater_than_equal_zero)), PARAM_LIST())

SFPU_INIT_CUSTOM_NAME(gez, greater_than_equal_zero)

}  // namespace ckernel
