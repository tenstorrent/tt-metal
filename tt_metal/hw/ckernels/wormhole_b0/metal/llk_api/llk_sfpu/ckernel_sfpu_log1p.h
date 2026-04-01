// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2015-2023 Norbert Juffa
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0 AND BSD-2-Clause

/*
 * The log1p(x) code is derived from code by Norbert Juffa.
 *
 * Copyright (c) 2015-2023, Norbert Juffa
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_log.h"

namespace ckernel {
namespace sfpu {

// For inputs with u = 1 + a > 0, write u = 2^k * t with t chosen in [0.75, 1.5),
// so m = t - 1 lies in [-0.25, 0.5). Then
//   log1p(a) = log(u) = k * log(2) + log1p(m).
// This implementation carries k in exponent-bit units (k << 23), which makes the
// rescaling and final k * log(2) term cheaper on SFPU.
// Inputs with 1 + a < 0 fall through to the default NaN.
// The boundary case u == 0 (a == -1) is outside the 2^k * t derivation above;
// it still evaluates to -inf via the same bit-level reduction path.
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat calculate_log1p_fp32(sfpi::vFloat a) {
    sfpi::vFloat u = a + sfpi::vConst1;
    sfpi::vFloat r = std::numeric_limits<float>::quiet_NaN();

    v_if(u >= 0.0f) {
        sfpi::vFloat three_quarters = 0.75f;
        sfpi::vInt e = sfpi::reinterpret<sfpi::vInt>(three_quarters);
        sfpi::vFloat e_float;

        // Subtracting the encoding of 0.75 and then zeroing the mantissa isolates
        // the exponent-bit offset e = k << 23 for the unique k with
        // 2^(-k) * u in [0.75, 1.5).
        e = sfpi::reinterpret<sfpi::vInt>(u) - e;
        e = sfpi::reinterpret<sfpi::vInt>(sfpi::setman(sfpi::reinterpret<sfpi::vFloat>(e), 0));

        // Reinterpreting a - e applies the same 2^(-k) scaling to a.
        // Affine correction below reconstructs
        //   m <- 2^(-k) * a + (2^(-k) - 1) = 2^(-k) * (1 + a) - 1.
        sfpi::vFloat m = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vInt>(a) - e);
        sfpi::vFloat neg_four = -4.0f;
        // Use s' = -4 * 2^(-k) instead of 4 * 2^(-k); see -0.25 for explanation.
        sfpi::vFloat s = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vInt>(neg_four) - e);

        // Use -0.25 (instead of 0.25) so we can reuse as polynomial coefficient later.
        sfpi::vFloat neg_quarter = -0.25f;
        sfpi::vFloat neg1 = sfpi::vConstNeg1;
        // t = -s' / 4 - 1 = 2^(-k) - 1
        sfpi::vFloat t = __builtin_rvtt_sfpmad(neg_quarter.get(), s.get(), neg1.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);

        // Minimax approximations for log1p(m) on [-0.25, 0.5]. Both paths keep the
        // exact linear term m explicit and approximate only the nonlinear
        // correction m^2 * P(m); the fp32 path keeps more terms than the
        // bf16-rounded path. fp16 or bf16 constants are used where possible to
        // reduce instruction count.
        if constexpr (is_fp32_dest_acc_en) {
            // log1p(x) = x + x*x * (
            //   -0x1p-1 + x * (0x1.555566p-2 + x * (-0x1p-2 + x * (0x1.998p-3 +
            //   x * (-0x1.55p-3 + x * (0x1.274p-3 + x * (-0x1.0c4p-3 + x *
            //   (0x1.b84p-4 + x * (-0x1.92cp-5)))))))))

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
            // log1p(x) = x + x*x * (-0x1.008p-1 + x * (0x1.744p-2 + x * (-0x1p-2)))

            sfpi::vInt abs_e = sfpi::abs(e);
            m = m + t;
            e_float = sfpi::int32_to_float(abs_e);

            s = m * m;
            r = neg_quarter * m + 0x1.744p-2f;
            r = r * m + -0x1.008p-1f;
        }
        // int32_to_float returns |e| as a real number in exponent-bit units;
        // restore sign and multiply by log(2) * 2^(-23) to recover k * log(2).
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
    // e represents k << 23 rather than k, so pre-fold the 2^(-23) factor into
    // the constant used for the final exponent contribution.
    sfpi::vConstFloatPrgm0 = LOG_TWO * TWO_TO_M23;

    if constexpr (is_fp32_dest_acc_en) {
        // Middle Horner coefficient used only by the fp32 polynomial.
        sfpi::vConstFloatPrgm1 = 0x1.555566p-2f;
    }
}

}  // namespace sfpu
}  // namespace ckernel
