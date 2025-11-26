// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_polyval.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool FAST_APPROX, bool HAS_BASE_SCALING, bool is_fp32_dest_acc_en = false>
sfpi_inline void calculate_log_body(const uint log_base_scale_factor) {
    ////////////////////////////
    // Load From dest + "normalize to calculation range"
    ////////////////////////////
    sfpi::vFloat in = sfpi::dst_reg[0];
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
    v_if(exp < 0) { exp = sfpi::setsgn(~exp + 1, 1); }
    v_endif;

    sfpi::vFloat expf = sfpi::int32_to_float(exp, 0);
    sfpi::vFloat vConstLn2 = sfpi::vConstFloatPrgm0;
    sfpi::vFloat result = expf * vConstLn2 + series_result;  // exp correction: ln(1+x) + exp*ln(2)

    if constexpr (HAS_BASE_SCALING) {
        result *= sfpi::s2vFloat16a(log_base_scale_factor);
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
        sfpi::vInt man = sfpi::exman9(in);
        sfpi::vInt signbit = sfpi::reinterpret<sfpi::vInt>(in) & 0x80000000;  // returns 0 for +ve value
        v_if((exp == 128 && man != 0) || in < 0.0F) {
            result = std::numeric_limits<float>::quiet_NaN();  // returns nan for fp32 and inf for bf16
        }
        v_elseif(signbit == 0 && exp == 128 && man == 0) { result = std::numeric_limits<float>::infinity(); }
        v_endif;
    }

    if constexpr (!is_fp32_dest_acc_en) {
        result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
    }

    sfpi::dst_reg[0] = result;
}

template <
    bool APPROXIMATION_MODE,
    bool FAST_APPROX,
    bool HAS_BASE_SCALING,
    bool is_fp32_dest_acc_en = false,
    int ITERATIONS = 8>
inline void calculate_log(uint log_base_scale_factor) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_log_body<FAST_APPROX, HAS_BASE_SCALING, is_fp32_dest_acc_en>(log_base_scale_factor);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool FAST_APPROX>
inline void log_init() {
    sfpi::vConstFloatPrgm0 = 0.693147182464599609375;  // ln(2)

    // XXXXX could do these to higher precision
    sfpi::vConstFloatPrgm1 = -2.0069785118103027;
    sfpi::vConstFloatPrgm2 = 3.767500400543213;
}

}  // namespace sfpu
}  // namespace ckernel
