// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

/*
 * The log(x) code is derived from code by Norbert Juffa.
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
#include "sfpu/ckernel_sfpu_polyval.h"

namespace ckernel {
namespace sfpu {

template <bool FAST_APPROX, bool HAS_BASE_SCALING, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat calculate_log_body(sfpi::vFloat a, const uint log_base_scale_factor) {
    sfpi::vFloat three_quarters = 0.75f;
    sfpi::vInt e = sfpi::reinterpret<sfpi::vInt>(a) - sfpi::reinterpret<sfpi::vInt>(three_quarters);

    if constexpr (!FAST_APPROX) {
        // normalise a (-0.0 and subnormals become +0.0)
        a = a * sfpi::vConst1 + sfpi::vConst0;
    }

    e = sfpi::reinterpret<sfpi::vInt>(sfpi::setman(sfpi::reinterpret<sfpi::vFloat>(e), 0));
    sfpi::vFloat m = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vInt>(a) - e);
    sfpi::vFloat result = std::numeric_limits<float>::quiet_NaN();

    // m in [0.75, 1.5). Compute log1p(m - 1) for m - 1 in [-0.25, 0.5).
    m -= sfpi::vConst1;

    v_if(a >= 0.0f) {
        sfpi::vFloat r;
        sfpi::vFloat s = m * m;
        sfpi::vFloat e_float;
        if constexpr (is_fp32_dest_acc_en) {
            r = -0x1.92cp-5f;
            r = r * m + 0x1.b84p-4f;
            r = r * m + -0x1.0c4p-3f;
            r = r * m + 0x1.274p-3f;
            r = r * m + -0x1.55p-3f;
            r = r * m + 0x1.998p-3f;
            sfpi::vMag abs_e = sfpi::abs(e);
            r = r * m + sfpi::vConstFloatPrgm1;
            e_float = sfpi::convert<sfpi::vFloat>(abs_e, sfpi::RoundMode::Nearest);
            r = r * m + sfpi::vConstFloatPrgm2;
            sfpi::vFloat neg_half = -0.5f;
            r = __builtin_rvtt_sfpmad(r.get(), m.get(), neg_half.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        } else {
            sfpi::vMag abs_e = sfpi::abs(e);
            sfpi::vFloat neg_quarter = -0.25f;
            r = neg_quarter * m + sfpi::vConstFloatPrgm1;
            e_float = sfpi::convert<sfpi::vFloat>(abs_e, sfpi::RoundMode::Nearest);
            r = r * m + sfpi::vConstFloatPrgm2;
        }

        // Handle special cases:
        //
        //   input 0.0  -> -inf
        //   input +inf -> +inf
        //   input NaN  -> NaN
        //
        // In the non-fast path, earlier normalisation maps -0.0 and subnormals
        // to +0.0. addexp(a, -1) wraps exponent 0 to 255, so zero becomes
        // +inf; exponent 255 values (Inf/NaN) are left unchanged.
        a = sfpi::addexp(a, -1);

        r = r * s + m;
        e_float = sfpi::copysgn(e_float, sfpi::reinterpret<sfpi::vFloat>(e));
        result = e_float * sfpi::vConstFloatPrgm0 + r;

        if constexpr (HAS_BASE_SCALING) {
            result *= sfpi::reinterpret<sfpi::vFloat>(sfpi::vUInt(log_base_scale_factor));
        }

        // For zero, result is negative before this multiply, so result * +inf
        // gives -inf. For +inf, result is positive, so result * +inf gives
        // +inf. NaNs either skip the main block or propagate here.
        v_if(sfpi::exexp(a, sfpi::ExponentMode::NoDebias) - 255 >= 0) { result *= a; }
        v_endif;
    }
    v_endif;

    return result;
}

template <
    bool APPROXIMATION_MODE,
    bool FAST_APPROX,
    bool HAS_BASE_SCALING,
    bool is_fp32_dest_acc_en,
    int ITERATIONS = 8>
inline void calculate_log(uint log_base_scale_factor) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat result = calculate_log_body<FAST_APPROX, HAS_BASE_SCALING, is_fp32_dest_acc_en>(
            sfpi::dst_reg[0], log_base_scale_factor);
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool FAST_APPROX, bool is_fp32_dest_acc_en>
inline void log_init() {
    const float LOG_TWO = 0.693147182f;       // 0x1.62e430p-1
    const float TWO_TO_M23 = 1.19209290e-7f;  // 0x1.0p-23
    // e represents k << 23 rather than k, so pre-fold the 2^(-23) factor into
    // the constant used for the final exponent contribution.
    sfpi::vConstFloatPrgm0 = LOG_TWO * TWO_TO_M23;

    if constexpr (is_fp32_dest_acc_en) {
        // Stored separately because the tuned fp32 m^3 and m^4 coefficients are
        // no longer the shared exact 1/3 and -1/4 values used in the bf16 path.
        sfpi::vConstFloatPrgm1 = -0x1.00001ap-2f;
        sfpi::vConstFloatPrgm2 = 0x1.555572p-2f;
    } else {
        // Horner coefficients used by bf16 polynomial
        sfpi::vConstFloatPrgm1 = 0x1.744p-2f;
        sfpi::vConstFloatPrgm2 = -0x1.008p-1f;
    }
}

}  // namespace sfpu
}  // namespace ckernel
