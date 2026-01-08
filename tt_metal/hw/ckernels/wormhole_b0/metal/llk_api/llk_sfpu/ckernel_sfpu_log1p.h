// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_log.h"

namespace ckernel {
namespace sfpu {

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat calculate_log1p_fp32(sfpi::vFloat a) {
    sfpi::vFloat u = a + sfpi::vConst1;
    sfpi::vFloat r = std::numeric_limits<float>::quiet_NaN();

    v_if(u >= 0.0f) {
        sfpi::vFloat three_quarters = 0.75f;
        sfpi::vInt e = sfpi::reinterpret<sfpi::vInt>(three_quarters);
        sfpi::vFloat e_float;

        e = sfpi::reinterpret<sfpi::vInt>(u) - e;
        e = sfpi::reinterpret<sfpi::vInt>(sfpi::setman(sfpi::reinterpret<sfpi::vFloat>(e), 0));

        sfpi::vFloat m = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vInt>(a) - e);
        sfpi::vFloat neg_four = -4.0f;
        sfpi::vFloat s =
            sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vInt>(neg_four) - e);  // s' in [2**-126,2**26]

        // t = 0.25f * s + sfpi::vConstNeg1;
        sfpi::vFloat neg_quarter = -0.25f;
        sfpi::vFloat neg1 = sfpi::vConstNeg1;
        sfpi::vFloat t = __builtin_rvtt_sfpmad(neg_quarter.get(), s.get(), neg1.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);

        // approximate log(1+m) on [-0.25, 0.5]
        if constexpr (is_fp32_dest_acc_en) {
            r = -0x1.92cp-5f;
            m = m + t;
            t = 0x1.b84p-4f;

            s = m * m;
            r = r * s + -0x1.0c4p-3f;
            t = t * s + 0x1.274p-3f;
            r = r * s + -0x1.55p-3f;
            t = t * s + 0x1.998p-3f;
            sfpi::vInt abs_e = sfpi::abs(e);
            r = r * s + neg_quarter;
            e_float = sfpi::int32_to_float(abs_e);
            r = t * m + r;
            r = r * m + sfpi::vConstFloatPrgm1;
            r = r * m + -0.5f;
        } else {
            sfpi::vInt abs_e = sfpi::abs(e);
            m = m + t;
            e_float = sfpi::int32_to_float(abs_e);

            s = m * m;
            r = 0x1.024p-3f;
            r = r * s + neg_quarter;
            r = r * m + -0.5f;
        }
        e_float = sfpi::setsgn(e_float, sfpi::reinterpret<sfpi::vFloat>(e));
        r = r * s + m;
        sfpi::vFloat infinity = std::numeric_limits<float>::infinity();
        r = e_float * sfpi::vConstFloatPrgm0 + r;

        // since u>=0, safely checks for u == NaN or u == inf
        v_if(sfpi::reinterpret<sfpi::vInt>(u) >= sfpi::reinterpret<sfpi::vInt>(infinity)) { r = u; }
        v_endif;
    }
    v_endif;

    return r;
}

/**
 * @tparam APPROXIMATION_MODE Ignored
 * @tparam FAST_APPROX Ignored
 * @tparam is_fp32_dest_acc_en If true, DEST registers are fp32, and output does not need to be rounded to bfloat16
 * @tparam ITERATIONS Number of iterations for given face
 */
template <bool APPROXIMATION_MODE, bool FAST_APPROX, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_log1p() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat result = calculate_log1p_fp32<is_fp32_dest_acc_en>(sfpi::dst_reg[0]);
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

/**
 * @tparam APPROXIMATION_MODE Ignored
 * @tparam FAST_APPROX Ignored
 * @tparam is_fp32_dest_acc_en If true, DEST registers are fp32, and output does not need to be rounded to bfloat16
 */
template <bool APPROXIMATION_MODE, bool FAST_APPROX, bool is_fp32_dest_acc_en>
inline void log1p_init() {
    const float LOG_TWO = 0.693147182f;       // 0x1.62e430p-1
    const float TWO_TO_M23 = 1.19209290e-7f;  // 0x1.0p-23
    sfpi::vConstFloatPrgm0 = LOG_TWO * TWO_TO_M23;

    if constexpr (is_fp32_dest_acc_en) {
        sfpi::vConstFloatPrgm1 = 0x1.555566p-2f;
    }
}

}  // namespace sfpu
}  // namespace ckernel
