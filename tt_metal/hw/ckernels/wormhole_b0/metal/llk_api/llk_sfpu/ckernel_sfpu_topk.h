// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_topk.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_bitonic_topk_phases_steps(
    uint idir, uint i_end_phase, uint i_start_phase, uint i_end_step, uint i_start_step) {
    _bitonic_topk_phases_steps<APPROXIMATION_MODE, ITERATIONS>(
        idir, i_end_phase, i_start_phase, i_end_step, i_start_step);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_bitonic_topk_merge(uint m_iter, uint k) {
    _bitonic_topk_merge<APPROXIMATION_MODE, ITERATIONS>(m_iter, k);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_bitonic_topk_rebuild(uint idir, uint m_iter, uint k, uint logk, uint skip_second) {
    _bitonic_topk_rebuild<APPROXIMATION_MODE, ITERATIONS>(idir, m_iter, k, logk, skip_second);
}

template <bool APPROXIMATION_MODE>
inline void topk_init() {
    _init_topk();
}

}  // namespace sfpu
}  // namespace ckernel
