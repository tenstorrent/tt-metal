// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpu/ckernel_sfpu_exp.h"
#include "llk_defs.h"

namespace ckernel {
namespace sfpu {

template <ApproximationMode APPROX_MODE, int ITERATIONS = 8>
inline void calculate_expm1() {
    const bool SCALE_EN = false;             // Expm1 does not use scale.
    const bool SKIP_POSITIVE_CHECK = false;  // Expm1 does not skip positive check.
    const uint16_t exp_base_scale_factor = p_sfpu::kCONST_1_FP16B;

        sfpi::vFloat d1 = sfpi::vFloat(sfpi::vConstFloatPrgm0);
        sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vConstIntPrgm1 + zif, 0);
        sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vConstIntPrgm2 + zif, 0);
        d2 = d1 * d2;
        zif = sfpu::_float_to_int32_(d2 * d3);

        // Restore exponent
        zii = sfpi::reinterpret<sfpi::vInt>(sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(zif), 127U + zii));

        y = sfpi::reinterpret<sfpi::vFloat>(zii) - sfpi::vConst1;
    }
    v_endif;
    if constexpr (!is_fp32_dest_acc_en) {
        // LRegs work on float32 data. If DST is bfloat16 then SFPSTORE will truncate it.
        // This can reduce accuracy: for instance, 9**2 = 80.8 gets round to 80.5
        // rather than 81 (which would have been correct).
        // To avoid this issue, we explicitly convert to bfloat16 using round-to-nearest-even.
        y = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(y, 0));
    }
    return y;
}

template <ApproximationMode APPROX_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_expm1() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        v = _calculate_exponential_piecewise_<(APPROX_MODE == ApproximationMode::Fast), SCALE_EN, SKIP_POSITIVE_CHECK>(
            v, exp_base_scale_factor);
        sfpi::dst_reg[0] = v - 1.0f;
        sfpi::dst_reg++;
    }
}

template <ApproximationMode APPROX_MODE>
void expm1_init() {
    const uint32_t EXP_BASE_SCALE_FACTOR = 0x3F800000;
    _init_exponential_<(APPROX_MODE == ApproximationMode::Fast), false /*fast_mode*/, EXP_BASE_SCALE_FACTOR>();
}

}  // namespace sfpu
}  // namespace ckernel
