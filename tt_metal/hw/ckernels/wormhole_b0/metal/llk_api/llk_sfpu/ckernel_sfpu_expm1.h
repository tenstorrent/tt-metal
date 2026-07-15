// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>

#include "ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_polyval.h"

/*
 * The expm1(x) code is derived from code by Norbert Juffa.
 *
 * Copyright (c) 2015-2023 Norbert Juffa
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

namespace ckernel::sfpu {

/*
 * i = rint(a / log(2)), f = a - i * log(2). Then
 * expm1(a) = 2**i * (expm1(f) + 1) - 1.
 *
 * Compute r = expm1(f). Then
 * expm1(a) = 2 * (0.5 * 2**i * r + 0.5 * 2**i - 0.5).
 *
 * With t = 0.5 * 2**i, expm1(a) = 2 * (r * t + t - 0.5).
 * For best accuracy, use expm1(a) = 2 * (r + 0.5) when i == 1,
 * and expm1(a) = r when i == 0.
 *
 * This approach avoids underflow for tiny values, and overflow for huge
 * values.
 */
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_expm1_(sfpi::vFloat a) {
    sfpi::vFloat log2e = sfpi::vConstFloatPrgm0;
    sfpi::vFloat rounding_bias = 12582912.f;
    sfpi::vFloat j = __builtin_rvtt_sfpmad(log2e.get(), a.get(), rounding_bias.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    sfpi::vFloat r;

    if constexpr (!is_fp32_dest_acc_en) {
        sfpi::vFloat scale, bias;

        r = 8.361816406e-03f;
        sfpi::vInt i = sfpi::as<sfpi::vInt>(j);
        j = j - rounding_bias;

        sfpi::vFloat c2 = 4.177856445e-02f;

        sfpi::vFloat f = j * sfpi::vConstFloatPrgm1 + a;  // -ln(2)

        r = r * f + c2;

        sfpi::vFloat s = f * f;

        r = r * f + sfpi::vConstFloatPrgm2;

        sfpi::vFloat w = 0.5f;
        r = __builtin_rvtt_sfpmad(r.get(), f.get(), w.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);

        sfpi::vFloat infinity = std::numeric_limits<float>::infinity();

        r = r * s + f;

        // For j == 0.0, r is already expm1(a). Avoid half-scaled
        // reconstruction as subnormals flush to zero, so
        // (0.5 * r) * 2 can lose tiny normal results.
        v_if(j != 0.0f) {
            sfpi::vFloat jm2 = j + -2.0f;
            // Keep reconstruction half-scaled: scale is 0.5 * 2**i. Avoids
            // materialising 2**i directly near overflow boundary.
            scale = sfpi::as<sfpi::vFloat>((i << 23) + sfpi::as<sfpi::vInt>(w));

            sfpi::vFloat abs_jm2 = sfpi::abs(jm2);
            bias = scale - w;
            sfpi::vInt tail = sfpi::as<sfpi::vInt>(sfpi::convert<sfpi::vSMag8>(abs_jm2, sfpi::RoundMode::Nearest));
            r = scale * r + bias;

            v_if(tail >= 127) {
                // Positive side becomes +inf; NaNs should propagate through the multiply.
                r = jm2 * infinity;

                v_if(jm2 < 0.0f) { r = -0.5f; }
                v_endif;
            }
            v_endif;
            r *= 2.0f;
        }
        v_endif;
    } else {
        sfpi::vFloat s, t, u, x, y;

        r = 1.974105835e-04f;
        sfpi::vInt i = sfpi::as<sfpi::vInt>(j);
        j = j - rounding_bias;

        sfpi::vFloat c4 = 1.393107930e-3f;

        sfpi::vFloat f = j * sfpi::vConstFloatPrgm1 + a;
        f = j * -1.42860677e-6f + f;  // -ln(2)_lo

        s = f * f;

        r = r * f + c4;
        r = r * f + 8.333439939e-3f;
        r = r * f + 4.166680202e-2f;
        sfpi::vFloat w = 0.5f;
        r = r * f + sfpi::vConstFloatPrgm2;
        sfpi::vFloat c0 = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(w) + -1);

        u = f;
        sfpi::vFloat jm1 = j + -1.0f;
        r = r * f + c0;
        v_if(jm1 == 0.0f) { u += 0.5f; }
        v_endif;
        r = r * s + u;

        v_if(j != 0.0f) {
            v_if(jm1 != 0.0f) {
                t = sfpi::as<sfpi::vFloat>((i << 23) + sfpi::as<sfpi::vInt>(w));
                y = t - w;
                sfpi::vFloat infinity = std::numeric_limits<float>::infinity();
                x = t - y;  // double-float canonicalization of difference
                sfpi::vFloat jm2 = jm1 + -1.0f;
                x = x - w;
                // abs(-NaN) = -NaN, otherwise the result will be positive.
                sfpi::vFloat abs_jm2 = sfpi::abs(jm2);
                r = r * t + x;
                // This will be -127 in the case of -NaN, otherwise 0 <= clamped <= 127.
                sfpi::vInt clamped =
                    sfpi::as<sfpi::vInt>(sfpi::convert<sfpi::vSMag8>(abs_jm2, sfpi::RoundMode::Nearest));
                r += y;
                // Handle special cases a * log2(e) <= -125 and a * log2(e) >= 129.
                v_if(clamped >= 127) {
                    // Positive case; multiply by j-2 to propagate NaN
                    r = jm2 * infinity;
                    v_if(jm2 < 0.0f) {
                        // Negative case; result will be -1 (note: -NaN was excluded earlier).
                        r = -0.5f;
                    }
                    v_endif;
                }
                v_endif;
            }
            v_endif;
            r *= 2.0f;
        }
        v_endif;
    }

    return r;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_expm1() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat y = _sfpu_expm1_<is_fp32_dest_acc_en>(x);
        if constexpr (!is_fp32_dest_acc_en) {
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = y;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void expm1_init() {
    sfpi::vConstFloatPrgm0 = 1.442695f;  // log2(e) == 1 / ln(2)
    if constexpr (is_fp32_dest_acc_en) {
        sfpi::vConstFloatPrgm1 = -0.693145752f;    // -ln(2)_hi
        sfpi::vConstFloatPrgm2 = 1.666667163e-1f;  // c1
    } else {
        sfpi::vConstFloatPrgm1 = -0.6931471805599453f;  // -ln(2)
        sfpi::vConstFloatPrgm2 = 1.666259766e-01f;      // c1
    }
}

}  // namespace ckernel::sfpu
