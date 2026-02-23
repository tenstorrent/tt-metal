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

template <ckernel::ApproximationMode APPROX_MODE, bool is_fp32_dest_acc_en, bool STABLE_SORT = false>
inline void calculate_bitonic_topk_phases_steps(
    uint idir, uint i_end_phase, uint i_start_phase, uint i_end_step, uint i_start_step) {
    _bitonic_topk_phases_steps<APPROX_MODE, is_fp32_dest_acc_en, STABLE_SORT>(
        idir, i_end_phase, i_start_phase, i_end_step, i_start_step);
}

template <ckernel::ApproximationMode APPROX_MODE, bool is_fp32_dest_acc_en, bool idir = false, bool STABLE_SORT = false>
inline void calculate_bitonic_topk_merge(uint m_iter, uint k) {
    _bitonic_topk_merge<APPROX_MODE, is_fp32_dest_acc_en, idir, STABLE_SORT>(m_iter, k);
}

template <ckernel::ApproximationMode APPROX_MODE, bool is_fp32_dest_acc_en, bool STABLE_SORT = false>
inline void calculate_bitonic_topk_rebuild(uint idir, uint m_iter, uint k, uint logk, uint skip_second) {
    _bitonic_topk_rebuild<APPROX_MODE, is_fp32_dest_acc_en, STABLE_SORT>(idir, m_iter, k, logk, skip_second);
}

template <ckernel::ApproximationMode APPROX_MODE>
inline void topk_init() {
    _init_topk();
}

}  // namespace sfpu
}  // namespace ckernel
