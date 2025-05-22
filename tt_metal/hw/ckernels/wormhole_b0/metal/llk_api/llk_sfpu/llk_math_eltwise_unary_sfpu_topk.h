// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_topk.h"

namespace ckernel {

// New LLK SFPU APIs

SFPU_INIT_CUSTOM_NAME_WITH_FN(topk, topk_local_sort, topk_init)

SFPU_CALCULATE_RC_CUSTOM(topk_local_sort, calculate_bitonic_topk_phases_steps, PARAM_LIST(), PARAM_LIST(PARAM(int, idir), PARAM(int, i_end_phase), PARAM(int, i_start_phase), PARAM(int, i_end_step), PARAM(int, i_start_step)))

SFPU_CALCULATE_RC_CUSTOM(topk_merge, calculate_bitonic_topk_merge, PARAM_LIST(DEFAULT_PARAM(bool, idir, false)), PARAM_LIST(PARAM(int, m_iter), PARAM(int, k)))

SFPU_CALCULATE_RC_CUSTOM(topk_rebuild, calculate_bitonic_topk_rebuild, PARAM_LIST(), PARAM_LIST(PARAM(bool, idir), PARAM(int, m_iter), PARAM(int, k), PARAM(int, logk), PARAM(int, skip_second)))

}  // namespace ckernel
