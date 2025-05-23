// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel {

// New LLK SFPU APIs

SFPU_CALCULATE_RC(reciprocal, calculate_reciprocal, PARAM_LIST(ARG(8), DEFAULT_PARAM(bool, is_fp32_dest_acc_en, false)), PARAM_LIST())

SFPU_INIT_CUSTOM_NAME_WITH_FN(reciprocal, reciprocal, recip_init)

}  // namespace ckernel
