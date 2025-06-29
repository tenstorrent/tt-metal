// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_topk.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

SFPU_INIT_KERNEL(topk, sfpu::topk_init)
SFPU_UNARY_PARAMS_KERNEL(
    topk_local_sort,
    RC_custom,
    int idir,
    int i_end_phase,
    int i_start_phase,
    int i_end_step,
    int i_start_step,
    idir,
    i_end_phase,
    i_start_phase,
    i_end_step,
    i_start_step)
SFPU_UNARY_PARAMS_KERNEL(topk_merge, RC_custom, int m_iter, int k, m_iter, k)
SFPU_UNARY_PARAMS_KERNEL(
    topk_rebuild,
    RC_custom,
    bool idir,
    int m_iter,
    int k,
    int logk,
    int skip_second,
    idir,
    m_iter,
    k,
    logk,
    skip_second)
