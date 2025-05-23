// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_log.h"

namespace ckernel {

// New LLK SFPU APIs

SFPU_INIT_WITH_FN(log)

SFPU_CALCULATE_RC(log, calculate_log, PARAM_LIST(ARG(false)), PARAM_LIST(ARG(0)))

SFPU_INIT_CUSTOM_NAME_WITH_FN(log_with_base, log_with_base, log_init)

SFPU_CALCULATE_RC(log_with_base, calculate_log, PARAM_LIST(ARG(true)), PARAM_LIST(PARAM(uint, base_scale)))

}  // namespace ckernel
