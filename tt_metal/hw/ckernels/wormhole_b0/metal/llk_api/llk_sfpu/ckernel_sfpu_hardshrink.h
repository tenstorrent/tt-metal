// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_row_pair.h"

namespace ckernel::sfpu {

inline void hardshrink_init() {
    // The two-row body advances Dest from its second store (see sfpu/ckernel_sfpu_row_pair.h).
    _init_sfpu_row_pair_();
    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_hardshrink(std::uint32_t param0) {
    // Hardshrink(x, λ) = x if |x| > λ, else 0
    // Single comparison against the magnitude: setsgn(v, 0) clears the sign bit.
    // param0 contains lambda as FP32 bits. For BF16 inputs, the host pre-rounds
    // lambda to BF16 precision (then re-expands to FP32) so that FP32→FP19b
    // truncation on SFPU preserves the BF16 value exactly. For FP32 inputs,
    // lambda is passed as full FP32, so both input and lambda undergo the same
    // FP32→FP19b truncation, keeping comparisons consistent.
    static_assert(ITERATIONS % 2 == 0, "calculate_hardshrink() processes dest rows in pairs.");
    sfpi::vFloat lambda = Converter::as_float(param0);
    // Two rows per body: the second row's store also advances Dest (see sfpu/ckernel_sfpu_row_pair.h).
    // The SFPMAD that forms each difference has a 2-cycle result shadow, and the compiler pads a
    // one-row body with an SFPNOP before the SFPSETCC; with two independent rows it interleaves the
    // second row's setsgn/MAD into that shadow instead. Each row runs the same operations as a
    // one-row body would, so the result is bit-identical to it.
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d += 2) {
        sfpi::vFloat v0 = sfpi::dst_reg[0];
        sfpi::vFloat v1 = sfpi::dst_reg[1];
        sfpi::vFloat abs_v0 = sfpi::setsgn(v0, 0 /* sgn */);
        sfpi::vFloat abs_v1 = sfpi::setsgn(v1, 0 /* sgn */);
        // Per row, abs_v <= lambda is written as d = lambda - abs_v >= 0. Wormhole SFPSETCC only
        // has LT0/GTE0/EQ0/NE0, so the <= form costs two SFPSETCCs plus an SFPCOMPC while this
        // one lands directly on GTE0. Both operands are non-negative (lambda is a magnitude), so
        // the subtraction cannot overflow and equality still lands on +0.
        //
        // abs_v comes from setsgn(v, 0) rather than sfpi::abs: SFPABS in float mode leaves a -NaN
        // sign-set (see ckernel_sfpu_isclose.h), whereas setsgn clears it unconditionally, so
        // abs() would feed a different operand into the compare for -NaN.
        sfpi::vFloat d0 = lambda - abs_v0;
        sfpi::vFloat d1 = lambda - abs_v1;
        v_if(d0 >= 0.0f) { sfpi::dst_reg[0] = 0.0f; }
        v_endif;
        v_if(d1 >= 0.0f) { _sfpu_row_pair_store_second_(0.0f); }
        v_endif;
        _sfpu_row_pair_end_();
    }
}

}  // namespace ckernel::sfpu
