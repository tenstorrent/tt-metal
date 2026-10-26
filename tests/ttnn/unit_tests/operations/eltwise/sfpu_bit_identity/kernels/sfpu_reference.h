// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The bit-identity test's reference kernels, in namespace ckernel::sfpu::ref: Blackhole's sigmoid / SiLU / exp as
// main computes them, one row per step through the single-row functions (_sfpu_sigmoid_, _sfpu_exp_accurate_) that
// other ops still use.

#pragma once

#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_sigmoid.h"

namespace ckernel::sfpu::ref {

template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_sigmoid() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat result = _sfpu_sigmoid_<is_fp32_dest_acc_en>(sfpi::dst_reg[0]);
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_silu() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat result = x * _sfpu_sigmoid_<is_fp32_dest_acc_en>(x);
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// sigmoid_init<false> and silu_init<false> on main: the reciprocal's Newton constant only.
inline void sigmoid_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<false>();
}

// calculate_exponential<false, true /*FP32 dest*/, false, ITERATIONS> on main.
template <int ITERATIONS>
inline void calculate_exponential_fp32() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::dst_reg[0] = _sfpu_exp_accurate_<true>(sfpi::dst_reg[0]);
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu::ref
