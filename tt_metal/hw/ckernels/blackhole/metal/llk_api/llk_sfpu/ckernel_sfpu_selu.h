// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_expm1_cw.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

// selu(x) = scale * x for x>=0, scale * alpha * (exp(x)-1) for x<0
// scale ≈ 1.0507, alpha ≈ 1.6733, scale*alpha ≈ 1.7581

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_selu(uint scale, uint alpha) {
    const sfpi::vFloat scale_val = Converter::as_float(scale);
    const sfpi::vFloat scale_alpha = Converter::as_float(scale) * Converter::as_float(alpha);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat result = scale_alpha * expm1_cw_clamped(x);

        v_if(x >= 0.0f) { result = scale_val * x; }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, sfpi::RoundMode::NearestEven));
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void selu_init() {}

}  // namespace ckernel::sfpu
