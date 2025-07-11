// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_erf_erfc.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

namespace ckernel {

// New LLK SFPU APIs

SFPU_ERF_ERFC_KERNEL(erf, erf)

SFPU_ERF_ERFC_KERNEL(erfc, erfc)

}  // namespace ckernel
