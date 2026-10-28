// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// One step of the diagonal complex LTI recurrence in one fp32 SFPU pass over DST:
//   x' = Lbar x + Bu   (complex, as real / imaginary tile pairs)
// Slots (tiles): 0 x_re, 1 x_im, 2 bu_re, 3 bu_im, 4 lbar_re, 5 lbar_im; x' goes to slots 0 and 1.
// first (t = 0): x' = Bu.

#if defined(TRISC_MATH)

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

namespace ckernel::sfpu {

template <bool first>
inline void calculate_lti_step() {
    constexpr uint32_t ts = 32;  // one DST tile in SFPU row addressing
#pragma GCC unroll 1
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat br = sfpi::dst_reg[2 * ts];
        sfpi::vFloat bi = sfpi::dst_reg[3 * ts];
        if constexpr (first) {
            sfpi::dst_reg[0] = br;
            sfpi::dst_reg[ts] = bi;
        } else {
            sfpi::vFloat xr = sfpi::dst_reg[0];
            sfpi::vFloat xi = sfpi::dst_reg[ts];
            sfpi::vFloat lr = sfpi::dst_reg[4 * ts];
            sfpi::vFloat li = sfpi::dst_reg[5 * ts];
            sfpi::dst_reg[0] = lr * xr - li * xi + br;
            sfpi::dst_reg[ts] = lr * xi + li * xr + bi;
        }
        sfpi::dst_reg++;
    }
}

inline void lti_step_init_fn() { math::reset_counters(p_setrwc::SET_ABD_F); }

}  // namespace ckernel::sfpu

namespace ckernel {

inline void llk_math_lti_step_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(ckernel::sfpu::lti_step_init_fn);
}

template <bool first>
inline void llk_math_lti_step() {
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::calculate_lti_step<first>, 0, VectorMode::RC);
}

}  // namespace ckernel

#endif  // TRISC_MATH
