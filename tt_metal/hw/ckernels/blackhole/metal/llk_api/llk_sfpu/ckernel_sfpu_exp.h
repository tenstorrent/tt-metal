// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_exp.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

sfpi_inline vFloat sfpu_exp(vFloat val) { return _sfpu_exp_(val); }

template <bool APPROXIMATION_MODE, bool SCALE_EN = false, int ITERATIONS = 8, bool FAST_APPROX = true>
void calculate_exponential(const uint iterations = ITERATIONS, const uint exp_base_scale_factor = 0) {
    _calculate_exponential_<APPROXIMATION_MODE, SCALE_EN, ITERATIONS, FAST_APPROX>(iterations, exp_base_scale_factor);
}

template <bool APPROXIMATION_MODE>
sfpi_inline vFloat calculate_exponential_body(vFloat in) {
    vFloat out;

    if constexpr (APPROXIMATION_MODE) {
        v_if(in >= 89) {
            vFloat val_inf = std::numeric_limits<float>::infinity();
            out = val_inf;
        }
        v_elseif(in < -42) { out = 0.0f; }
        v_else { out = _calculate_exponential_body_<APPROXIMATION_MODE>(in); }
        v_endif;
    } else {
        out = _calculate_exponential_body_<APPROXIMATION_MODE>(in);
    }

    return out;
}

template <bool APPROXIMATION_MODE>
sfpi_inline vFloat calculate_exponential_body_improved(vFloat val) {
    vFloat out;
    if constexpr (APPROXIMATION_MODE) {
        v_if(val >= 89) {
            vFloat val_inf = std::numeric_limits<float>::infinity();
            out = val_inf;
        }
        v_elseif(val < -42) { out = 0.0f; }
        v_else {
            // * by 1/ln2 and add convert to 7.3 FxP format
            vFloat vConstLn2Recip = vConstFloatPrgm0;
            vFloat c23_73 = vConstFloatPrgm1;
            vInt adj_exp = vConstIntPrgm2;
            val = val * vConstLn2Recip + c23_73;

            // Remove Exponent of 7 and bias the Mantissa to 127.
            vInt val_short = adj_exp + reinterpret<vInt>(val);

            // SHL to move integer bits to exponent
            val_short <<= 10 - p_exp::FRAC_BITS;
            out = reinterpret<vFloat>(val_short);
        }
        v_endif;
    } else {
        // Force sign to 0 (make number positive)
        out = sfpu_exp(setsgn(val, 0));
        v_if(val < 0) { out = sfpu_reciprocal(out); }
        v_endif;
    }
    return out;
}

template <bool APPROXIMATION_MODE, bool FAST_APPROX = true>
void exp_init() {
    _init_exponential_<APPROXIMATION_MODE, FAST_APPROX>();
}

}  // namespace sfpu
}  // namespace ckernel
