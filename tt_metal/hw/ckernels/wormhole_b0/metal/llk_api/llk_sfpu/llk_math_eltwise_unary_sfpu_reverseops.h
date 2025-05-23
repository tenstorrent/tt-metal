// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_reverseops.h"

namespace ckernel {

/************** rsub ************/

SFPU_INIT_CUSTOM_NAME_WITH_FN(rsub, unused, rsub_init)

SFPU_CALCULATE_ALWAYS_RC(rsub, calculate_rsub, PARAM_LIST(ARG(8)), PARAM_LIST(DEFAULT_PARAM(uint, param0, 0)))

}  // namespace ckernel
