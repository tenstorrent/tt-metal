// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_isfinite() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        v_if(
            v == std::numeric_limits<float>::infinity() || v == -std::numeric_limits<float>::infinity() ||
            v == std::numeric_limits<float>::quiet_NaN() || v == std::numeric_limits<float>::signaling_NaN()) {
            v = 0.0f;
        }
        v_else { v = 1.0f; }
        v_endif;

        dst_reg[0] = v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_isinf() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat in = dst_reg[0];
        sfpi::vInt exp = sfpi::exexp(in);
        sfpi::vInt man = sfpi::exman9(in);
        vFloat out = sfpi::vConst0;
        v_if(exp == 128 && man == 0) { out = sfpi::vConst1; }
        v_endif;
        dst_reg[0] = out;
        dst_reg++;
    }
}

/* Checks if the sign bit of the floating point number in DEST
is positive. Checks if the exponent is 128 and mantissa is 0.
If all of the three conditions are met, the number is marked as
positive infinity, so '1' is written in the location of the DEST
where the number was stored. Otherwise, `0` is written instead
of the number.
*/
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_isposinf() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat in = dst_reg[0];
        sfpi::vInt exp = sfpi::exexp(in);
        sfpi::vInt man = sfpi::exman9(in);
        vFloat out = sfpi::vConst0;
        vInt signbit = sfpi::reinterpret<sfpi::vInt>(in) & 0x80000000;  // returns 0 for +ve value
        v_if(signbit == 0 && exp == 128 && man == 0) { out = sfpi::vConst1; }
        v_endif;
        dst_reg[0] = out;
        dst_reg++;
    }
}

/* Checks if the sign bit of the floating point number in DEST
is negative. Checks if the exponent is 128 and mantissa is 0.
If all of the three conditions are met, the number is marked as
negative infinity, so '1' is written in the location of the DEST
where the number was stored. Otherwise, `0` is written instead
of the number.
*/
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_isneginf() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat in = dst_reg[0];
        sfpi::vInt exp = sfpi::exexp(in);
        sfpi::vInt man = sfpi::exman9(in);
        vFloat out = sfpi::vConst0;
        vInt signbit = sfpi::reinterpret<sfpi::vInt>(in) & 0x80000000;  // returns 0 for +ve value
        v_if(signbit == 0x80000000 && exp == 128 && man == 0) { out = sfpi::vConst1; }
        v_endif;
        dst_reg[0] = out;
        dst_reg++;
    }
}

/* Checks if the exponent is 128 and mantissa is not 0.
If both conditions are met, the number is marked as
nan, so '1' is written in the location of the DEST
where the number was stored. Otherwise, `0` is written instead
of the number.
*/
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_isnan() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat in = dst_reg[0];
        sfpi::vInt exp = sfpi::exexp(in);
        sfpi::vInt man = sfpi::exman9(in);
        vFloat out = sfpi::vConst0;
        v_if(exp == 128 && man != 0) { out = sfpi::vConst1; }
        v_endif;
        dst_reg[0] = out;
        dst_reg++;
    }
}

template <SfpuType operation, bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_sfpu_isinf_isnan() {
    if constexpr (operation == SfpuType::isinf) {
        calculate_isinf<APPROXIMATION_MODE, ITERATIONS>();
    } else if constexpr (operation == SfpuType::isposinf) {
        calculate_isposinf<APPROXIMATION_MODE, ITERATIONS>();
    } else if constexpr (operation == SfpuType::isneginf) {
        calculate_isneginf<APPROXIMATION_MODE, ITERATIONS>();
    } else if constexpr (operation == SfpuType::isnan) {
        calculate_isnan<APPROXIMATION_MODE, ITERATIONS>();
    } else if constexpr (operation == SfpuType::isfinite) {
        calculate_isfinite<APPROXIMATION_MODE, ITERATIONS>();
    }
}

}  // namespace sfpu
}  // namespace ckernel
