// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "noc_nonblocking_api.h"

#include "sfpi.h"

using namespace sfpi;

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _bitonic_topk_phases_steps(
    const int idir, const int i_end_phase, const int i_start_phase, const int i_end_step, const int i_start_step)
{
    TT_LLK_DUMP("_bitonic_topk_phases_steps()");
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _bitonic_topk_merge(const int m_iter, const int k)
{
    TT_LLK_DUMP("_bitonic_topk_merge()");
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _bitonic_topk_rebuild(const bool idir, const int m_iter, const int k, const int logk, const int skip_second)
{
    TT_LLK_DUMP("_bitonic_topk_rebuild()");
}

inline void _init_topk()
{
    TT_LLK_DUMP("_init_topk()");
}

} // namespace sfpu
} // namespace ckernel
