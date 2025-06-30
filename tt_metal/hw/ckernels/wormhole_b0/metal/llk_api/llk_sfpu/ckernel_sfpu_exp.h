// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "noc_nonblocking_api.h"
#include "sfpu/ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include <limits>

#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

sfpi_inline vFloat sfpu_exp(vFloat val) { return _sfpu_exp_(val); }

sfpi_inline sfpi::vFloat _sfpu_exp_21f_(sfpi::vFloat val) {
    sfpi::vInt z = sfpu::_float_to_int32_(val * sfpi::vFloat(0x00b8aa3b) + sfpi::vFloat(0x3f800000));
    sfpi::vInt zii = z & 0x7f800000;
    sfpi::vInt zif = z & sfpi::vInt(0x007fffff);  // extra mantissa

    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7);
    sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + zif);
    sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560) + zif);
    d2 = d1 * d2;
    zif = sfpu::_float_to_int32_(d2 * d3);

    zii |= zif;  // restore exponent

    sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(zii);

    return y;
}

sfpi_inline sfpi::vFloat _sfpu_exp_21f_alt_(sfpi::vFloat val) {
    // sfpi::vFloat val_debiased = val * sfpi::vConstFloatPrgm2;
    // val_debiased = addexp(val_debiased, 127);
    // sfpi::vInt z = sfpu::_float_to_int32_(val_debiased);
    // return val;

    sfpi::vFloat val_debiased = val * sfpi::vConstFloatPrgm2 + sfpi::vFloat(0x3f800000);
    // val_debiased = addexp(val_debiased, 127);
    sfpi::vInt z = sfpu::_float_to_int32_(val_debiased);

    sfpi::vInt zii = z & 0x7f800000;
    sfpi::vInt zif = z & sfpi::vInt(0x007fffff);  // extra mantissa

    sfpi::vFloat d1 = sfpi::s2vFloat16b(0.4027970135211944580078125e-7);

    sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + zif);
    sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560) + zif);
    d2 = d1 * d2;
    zif = sfpu::_float_to_int32_(d2 * d3);

    zii |= zif;  // restore exponent

    sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(zii);

    return y;
}

template <
    bool APPROXIMATION_MODE,
    bool FAST_APPROX,
    bool ACCURATE,
    bool SCALE_EN = false,
    int ITERATIONS = 8,
    bool SKIP_POSITIVE_CHECK = false>
void calculate_exponential(const uint iterations = ITERATIONS, const uint exp_base_scale_factor = 0x3F80) {
    if constexpr (ACCURATE) {
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];
            sfpi::vFloat result = _sfpu_exp_21f_(val);
            // sfpi::vFloat result = 5.0;
            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    } else if constexpr (!ACCURATE) {
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];
            sfpi::vFloat result = _sfpu_exp_21f_alt_(val);
            // sfpi::vFloat result = 3.0;
            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    } else {
        _calculate_exponential_<APPROXIMATION_MODE, SCALE_EN, ITERATIONS, FAST_APPROX, SKIP_POSITIVE_CHECK>(
            iterations, exp_base_scale_factor);
    }
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

template <bool APPROXIMATION_MODE, bool FAST_APPROX, bool ACCURATE, uint32_t scale = 0x3F800000>
void exp_init() {
    if constexpr (!ACCURATE) {
        sfpi::vConstFloatPrgm0 = 1.442695f;  // ln2_recip
        sfpi::vConstFloatPrgm1 = 2.0f;
        sfpi::vConstFloatPrgm2 = 0x00b8aa3b;
    } else {
        _init_exponential_<APPROXIMATION_MODE, FAST_APPROX, scale>();
    }
}

}  // namespace sfpu
}  // namespace ckernel
