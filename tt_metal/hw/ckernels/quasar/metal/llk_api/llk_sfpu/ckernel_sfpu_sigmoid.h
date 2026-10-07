// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_sigmoid_appx.h"
#include "sfpu/ckernel_sfpu_sigmoid.h"

namespace ckernel {
namespace sfpu {

// Blackhole's SFPI sigmoid body and its init (clamped_silu_glu calls them), copied from the Blackhole
// metal ckernel_sfpu_sigmoid.h; calculate_sigmoid below stays Quasar's kernel.
template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline sfpi::vFloat _sfpu_sigmoid_(sfpi::vFloat x) {
    // Compute sigmoid as:
    // sigmoid(x) = 1 / (1 + exp(-x))

    sfpi::vFloat exp_neg_x;
    // If fp32 then use higher accuracy exp function
    // Otherwise, use exp_21f (~1 ULP on bfloat16)
    if constexpr (is_fp32_acc_to_dest_mode) {
        exp_neg_x = _sfpu_exp_accurate_<true>(-x);
    } else {
        exp_neg_x = _sfpu_exp_21f_bf16_<true>(-x);
    }

    sfpi::vFloat denominator = 1.0f + exp_neg_x;

    sfpi::vFloat result;
    if constexpr (is_fp32_acc_to_dest_mode) {
        result = _sfpu_reciprocal_<2>(denominator);
    } else {
        result = _sfpu_reciprocal_<1>(denominator);
    }

    return result;
}

template <bool APPROXIMATION_MODE>
inline void sigmoid_init() {
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
    if constexpr (!APPROXIMATION_MODE) {
        _init_reciprocal_<false>();
    } else {
        sigmoid_appx_init();
    }
}

template <int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_sigmoid() {
    _calculate_sigmoid_<ITERATIONS>();
}

}  // namespace sfpu
}  // namespace ckernel
