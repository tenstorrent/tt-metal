// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_log.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

namespace ckernel {

// New LLK SFPU APIs

SFPU_INIT_LITERAL_KERNEL(log, sfpu::log_init, 0)

SFPU_INIT_AND_UNARY_PARAMS_KERNEL_WITH_EXTRA_TEMPLATE_ARG(
    log_with_base,                 // OP
    log_with_base,                 // TYPE
    sfpu::log_init,                // INIT_CB
    RC,                            // MODE
    ckernel::sfpu::calculate_log,  // CALC_CB
    true,                          // EXTRA_TEMPLATE (base_flag = true)
    uint base_scale,               // EXTRA_ARG_DECL
    base_scale)                    // EXTRA_ARG_PASS

}  // namespace ckernel
