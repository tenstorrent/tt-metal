// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "noc_nonblocking_api.h"

#include "sfpi.h"

using namespace sfpi;

namespace ckernel
{
namespace sfpu
{

template <bool HAS_BASE_SCALING>
sfpi_inline void _calculate_log_body_(const int log_base_scale_factor)
{
    ////////////////////////////
    // Load From dest + "normalize to calculation range"
    ////////////////////////////
    vFloat in = dst_reg[0];
    vFloat x = setexp(in, 127);    // set exp to exp bias (put in range of 1-2)

    ////////////////////////////
    // Calculate Cheby Approximation using Horner Form Multiplication: 3rd Order
    // x* ( x* (A*x + B) + C) + D
    // A :0.1058, B: -0.3942, C: 0.9813, D: 0.006
    // Run above on (x-1) so x is in ln(x+1), plug (x-1 into equation above to
    // save the subtract and get A',B',C',D':
    // A' = A
    // B' = -3A + B
    // C' = 3a -2B + C
    // D' = -A + B - C + D
    // A':0.1058, B':-0.7116, C':2.0871, D':-1.4753
    ////////////////////////////
    vFloat a = s2vFloat16a(0.1058F);
    vFloat series_result = x * (x * (x * a + s2vFloat16a(-0.7122f)) + s2vFloat16a(2.0869)) + s2vFloat16a(-1.4753f);

    ////////////////////////////
    // Convert exponent to float
    ////////////////////////////
    // Extract exponent and calculate abs value.  Save sign into partial reg
    vInt exp = 0;
    v_if (in != 0.0F) {
        exp = exexp(in);
        v_if (exp < 0) {
            exp = sfpi::abs(exp);
            in = setsgn(in, 1);
        }
        v_endif;
    }
    v_endif;

    // Calculate exponent of the exponent value. Done by using LZ
    // Get leading zero.  If not zero, we do 19 + ~LZ to get exponent value (mathematically == 19 - LZ - 1)
    vInt new_exp = 0;
    v_if (exp != 0) {
        new_exp = lz(exp);
        new_exp = ~new_exp;
        new_exp += 19;
        v_if (new_exp >= 0) {
            new_exp += 127;
        }
        v_endif;
    }
    v_endif;

    vFloat result = setexp(in, new_exp);
    vInt shift = lz(exp) + 1;
    result = setman(result, shft(reinterpret<vUInt>(exp), shift));
    result = result * vConst0p6929 + series_result; // exp correction: ln(1+x) + exp*ln(2)

    if constexpr (HAS_BASE_SCALING) {
        result *= s2vFloat16a(log_base_scale_factor);
    }

    ////////////////////////////
    // Base case when input is 0. ln(0) = -inf
    ////////////////////////////
    v_if (dst_reg[0] == 0.0F) { // Reload for register pressure
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    dst_reg[0] = result;
}

template <bool APPROXIMATION_MODE, bool HAS_BASE_SCALING, int ITERATIONS>
inline void _calculate_log_(uint log_base_scale_factor)
{
    #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        _calculate_log_body_<HAS_BASE_SCALING>(log_base_scale_factor);
        dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
