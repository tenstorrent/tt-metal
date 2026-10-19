// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <int ITERATIONS = 8>
inline void calculate_sigmoid_appx() {
    sfpi::vLut8si si0 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vLut8si si1 = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vLut8si si2 = sfpi::l_reg[sfpi::LRegs::LReg2];

    // abs_bits == inf_bits -> +/-Inf. NaN's non-zero mantissa misses the compare, the
    // same classification ckernel_sfpu_binary.h and ckernel_sfpu_isclose.h rely on.
    constexpr int32_t inf_bits = 0x7F800000;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat result = sfpi::lut(val, si0, si1, si2) + 0.5f;

        // Specials: the top LUT segment is (slope 0.0f, intercept 0.5f) and the hardware
        // evaluates A * |x| + B, so x = +/-inf reaches that segment as 0 * inf + 0.5 = NaN
        // -- the same failure approximate tanh records for its (0, 1) top segment
        // (tt-llk#1701 item 18). IEEE/torch sigmoid is +inf -> 1.0, -inf -> 0.0, NaN ->
        // NaN. NaN lanes miss the predicate and keep the LUT's NaN. copysgn(1.0f, val) is
        // +1.0f on +inf and -1.0f on -inf, and max(..., 0.0f) maps the negative arm to
        // 0.0f -- both exact, and no other value crosses the predicate. vConstIntPrgm0
        // holds the abs mask, programmed in sigmoid_appx_init.
        sfpi::vInt abs_bits = sfpi::as<sfpi::vInt>(val) & sfpi::vConstIntPrgm0;
        v_if (abs_bits == inf_bits) {
            result = sfpi::max(sfpi::copysgn(1.0f, val), 0.0f);
        }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }

    sfpi::l_reg[sfpi::LRegs::LReg0] = si0;
    sfpi::l_reg[sfpi::LRegs::LReg1] = si1;
    sfpi::l_reg[sfpi::LRegs::LReg2] = si2;
}

inline void sigmoid_appx_init() {
    // Load the 3 fp16b LUT coefficients into LReg0-2
    // Abs mask for the specials guard in calculate_sigmoid_appx; programming it here
    // keeps the per-element loop free of the rebuilt-immediate SFPLOADIs a float-literal
    // mask would cost (the same reason ckernel_sfpu_isclose.h does this).
    sfpi::vConstIntPrgm0 = 0x7FFFFFFF;
    sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vLut8si(0.22656f, 0.0f);
    sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vLut8si(0.26562f, -0.04687f);
    sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vLut8si(0.0f, 0.5f);
}

}  // namespace sfpu
}  // namespace ckernel
