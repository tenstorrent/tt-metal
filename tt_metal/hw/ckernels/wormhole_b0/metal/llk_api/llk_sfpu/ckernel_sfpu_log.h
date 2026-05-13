// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * The fp32 log(x) code is derived from code by Norbert Juffa.
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
sfpi_inline sfpi::vFloat calculate_log_body(sfpi::vFloat in, const uint log_base_scale_factor) {
    ///////////////////////////////////
    // "normalize to calculation range"
    ///////////////////////////////////
    sfpi::vFloat x = sfpi::setexp(in, 127);  // set exp to exp bias (put in range of 1-2)

    // Minimax approximation of log(x) over [1; 2] calculated using Sollya with the following command:
    // > fpminimax(log(x), 5, [|single...|], [1+2^(-20); 2], relative);
    sfpi::vFloat series_result = PolynomialEvaluator::eval(
        x,
        sfpi::vConstFloatPrgm1,
        sfpi::vConstFloatPrgm2,
        -2.800232410430908,
        1.3681391477584839,
        -0.3706687390804291,
        0.04224011301994324);

    ////////////////////////////
    // Convert exponent to float
    ////////////////////////////
    sfpi::vInt exp = sfpi::exexp(in);

    // Convert negative numbers: signed -> sign-magnitude
    v_if(exp < 0) { exp = sfpi::setsgn(~exp + 1, 1); }
    v_endif;

    sfpi::vFloat expf = sfpi::int32_to_float(exp, sfpi::RoundMode::NearestEven);
    sfpi::vFloat vConstLn2 = sfpi::vConstFloatPrgm0;
    sfpi::vFloat result = expf * vConstLn2 + series_result;  // exp correction: ln(1+x) + exp*ln(2)

    if constexpr (HAS_BASE_SCALING) {
        result *= sfpi::reinterpret<sfpi::vFloat>(sfpi::vUInt(log_base_scale_factor));
    }

    ////////////////////////////
    // Base case when input is 0. ln(0) = -inf
    ////////////////////////////
    v_if(in == 0.0F) {  // Reload for register pressure
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    if constexpr (!FAST_APPROX) {
        sfpi::vInt exp = sfpi::exexp(in);
        v_if(sfpi::reinterpret<sfpi::vInt>(in) == 0x7F800000) {
            // If input is infinity, return infinity
            result = std::numeric_limits<float>::infinity();
        }
        v_elseif(exp == 128 || in < 0.f) {                     // +inf or negative input -> NaN
            result = std::numeric_limits<float>::quiet_NaN();  // returns nan for fp32 and inf for bf16
        }
        v_endif;
    }

    if constexpr (!is_fp32_dest_acc_en) {
        result = sfpi::float_to_fp16b(result, sfpi::RoundMode::NearestEven);
    }

    return result;
}

template <bool HAS_BASE_SCALING>
sfpi_inline sfpi::vFloat calculate_log_f32_body(sfpi::vFloat a, const uint log_base_scale_factor) {
    sfpi::vFloat result = std::numeric_limits<float>::quiet_NaN();
    sfpi::vFloat infinity = std::numeric_limits<float>::infinity();

    sfpi::vFloat two_thirds = 0.666666667f;
    sfpi::vInt u = sfpi::reinterpret<sfpi::vInt>(a) - sfpi::reinterpret<sfpi::vInt>(two_thirds);
    sfpi::vInt e = sfpi::exexp_nodebias(sfpi::reinterpret<sfpi::vFloat>(u));
    sfpi::vFloat m = sfpi::reinterpret<sfpi::vFloat>(sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(u)));

    sfpi::vFloat e_float = sfpi::int32_to_float(e, sfpi::RoundMode::NearestEven);
    e_float = sfpi::copysgn(e_float, sfpi::reinterpret<sfpi::vFloat>(u));

    // m in [2/3, 4/3]. Compute log1p(m - 1) for m - 1 in [-1/3, 1/3].
    m -= sfpi::vConst1;
    sfpi::vFloat s = m * m;
    sfpi::vFloat r = -0.130310059f;
    sfpi::vFloat t = 0.140869141f;
    r = r * s + -0.121486895f;
    t = t * s + 0.139815226f;
    r = r * s + -0.166845486f;
    t = t * s + 0.200120315f;
    r = r * s + -0.249996230f;
    r = t * m + r;
    r = r * m + 0.333331972f;
    r = r * m + -0.500000000f;
    r = r * s + m;
    result = e_float * sfpi::vConstFloatPrgm0 + r;

    if constexpr (HAS_BASE_SCALING) {
        result *= sfpi::reinterpret<sfpi::vFloat>(sfpi::vUInt(log_base_scale_factor));
    }

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
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat result;
        if constexpr (!is_fp32_dest_acc_en) {
            result = calculate_log_body<FAST_APPROX, HAS_BASE_SCALING, is_fp32_dest_acc_en>(in, log_base_scale_factor);
        } else {
            result = calculate_log_f32_body<HAS_BASE_SCALING>(in, log_base_scale_factor);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool FAST_APPROX, bool is_fp32_dest_acc_en>
inline void log_init() {
    if constexpr (!is_fp32_dest_acc_en) {
        sfpi::vConstFloatPrgm0 = 0.693147182464599609375;  // ln(2)
        sfpi::vConstFloatPrgm1 = -2.0069785118103027;
        sfpi::vConstFloatPrgm2 = 3.767500400543213;
    } else {
        constexpr float LOG_TWO = 0.693147182f;       // 0x1.62e430p-1
        constexpr float TWO_TO_M23 = 1.19209290e-7f;  // 0x1.0p-23
        // e represents k << 23, so pre-fold 2^(-23) into the exponent contribution.
        sfpi::vConstFloatPrgm0 = LOG_TWO * TWO_TO_M23;
    }
}

}  // namespace sfpu
}  // namespace ckernel
