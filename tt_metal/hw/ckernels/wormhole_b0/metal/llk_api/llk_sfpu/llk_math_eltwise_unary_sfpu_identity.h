// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_identity.h"

namespace ckernel {

// New LLK SFPU APIs

SFPU_CALCULATE_RC(identity, calculate_identity, PARAM_LIST(ARG(8)), PARAM_LIST())

SFPU_CALCULATE_RC(identity_uint32, calculate_identity_uint, PARAM_LIST(ARG(8)), PARAM_LIST())

SFPU_INIT_CUSTOM_NAME(identity, unused)

}  // namespace ckernel
