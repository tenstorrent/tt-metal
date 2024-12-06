// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "ckernel_sfpu_exp.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel
{
namespace sfpu
{

enum {
    ADD_BINARY = 0,
    SUB_BINARY = 1,
    MUL_BINARY = 2,
    DIV_BINARY = 3,
    RSUB_BINARY = 4,
    POW_BINARY = 5
};  // BINOP_MODE

sfpi_inline vFloat _calculate_sfpu_binary_power_(vFloat base, vFloat pow)
{
    vFloat original_base = base;

    // Check for integer power
    vInt pow_int = float_to_int16(pow, 0); // int16 should be plenty, since large powers will approach 0/Inf
    vFloat pow_rounded = int32_to_float(pow_int, 0);
    v_if (pow_rounded == pow) {
        // if pow is integer, set base to positive
        base = setsgn(base, 0);
    }
    v_endif;

    // Normalize base to calculation range
    vFloat x = setexp(base, 127);    // set exp to exp bias (put base in range of 1-2)

    // 3rd order polynomial approx - determined using rminimax over [1,2]
    vFloat series_result = x * (x * (x * 0x2.44734p-4f - 0xd.e712ap-4f) + 0x2.4f5388p+0f) - 0x1.952992p+0f;

    // Convert exponent to float
    vInt exp = exexp(base);
    v_if (exp < 0) {
        exp = setsgn(~exp + 1, 1);
    }
    v_endif;
    vFloat expf = int32_to_float(exp, 0);
 
    // De-normalize to original range
    vFloat vConstLn2 = 0.692871f;
    vFloat log_result = expf * vConstLn2 + series_result; // exp correction: ln(1+x) + exp*ln(2)

    // Base case when input is 0. ln(0) = -inf
    v_if (base == 0.0f) { // Reload for register pressure
        log_result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    // Take exp(pow * log(base)) to produce base^pow
    vFloat val = pow * log_result;

    // Force sign to 0 (make number positive)
    vFloat result = _sfpu_exp_(setsgn(val, 0));

    v_if (val < 0) {
        result = _sfpu_reciprocal_(result);
    }
    v_endif;

    // Check valid base range
    v_if (original_base < 0.0f) { // negative base
        // Check for integer power
        v_if (pow_rounded == pow) {
            // if pow is odd integer, set result to negative
            v_if (pow_int & 0x1) {
                result = setsgn(result, 1);
            }
            v_endif;
        } v_else {
            result = std::numeric_limits<float>::quiet_NaN();
        }
        v_endif;
    }
    v_endif;

    return result;
}

template <bool APPROXIMATION_MODE, int BINOP_MODE, int ITERATIONS = 8>
inline void _calculate_sfpu_binary_(const uint dst_offset)
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size = 32;
        vFloat in0 = dst_reg[0];
        vFloat in1 = dst_reg[dst_offset * dst_tile_size];
        vFloat result = 0.0f;

        if constexpr (BINOP_MODE == ADD_BINARY) {
            result = in0 + in1;
        } else if constexpr (BINOP_MODE == SUB_BINARY) {
            result = in0 - in1;
        } else if constexpr (BINOP_MODE == MUL_BINARY) {
            result = in0 * in1;
        } else if constexpr (BINOP_MODE == DIV_BINARY) {
            v_if (in1 == 0) {
                v_if (in0 == 0) {
                    result = std::numeric_limits<float>::quiet_NaN();
                } v_else {
                    result = std::numeric_limits<float>::infinity();
                    result = setsgn(result, in0);
                }
                v_endif;
            } v_elseif (in0 == in1) {
                result = vConst1;
            } v_else {
                result = in0 * setsgn(_sfpu_reciprocal_<4>(in1), in1);
            }
            v_endif;
        } else if constexpr (BINOP_MODE == RSUB_BINARY) {
            result = in1 - in0;
        } else if constexpr (BINOP_MODE == POW_BINARY) {
            result = _calculate_sfpu_binary_power_(in0, in1);
        }

        dst_reg[0] = result;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
