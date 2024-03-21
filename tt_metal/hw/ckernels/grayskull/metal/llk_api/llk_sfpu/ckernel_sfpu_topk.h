// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "ckernel_sfpu_topk.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// calculate_bitonic_topk_phases_steps is unused for Grayskull
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_bitonic_topk_phases_steps()
{
    _bitonic_topk_phases_steps<APPROXIMATION_MODE, ITERATIONS>(0, 0, 0, 0, 0);
}

// calculate_bitonic_topk_merge is unused for Grayskull
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_bitonic_topk_merge()
{
    _bitonic_topk_merge<APPROXIMATION_MODE, ITERATIONS>(0, 0);
}

// calculate_bitonic_topk_rebuild is unused for Grayskull
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_bitonic_topk_rebuild()
{
    _bitonic_topk_rebuild<APPROXIMATION_MODE, ITERATIONS>(0, 0, 0, 0, 0);
}

// topk_init is unused for Grayskull
template <bool APPROXIMATION_MODE>
inline void topk_init()
{
    _init_topk();
}

}  // namespace sfpu
}  // namespace ckernel
