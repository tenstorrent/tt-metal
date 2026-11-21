// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "ckernel_sfpu_row_pair.h"

namespace ckernel::sfpu {

inline void hardshrink_init() {
    // The two-row body advances Dest from its second store (see ckernel_sfpu_row_pair.h).
    _init_sfpu_row_pair_();
    math::reset_counters(p_setrwc::SET_ABD_F);
}

/**
 * @brief hardshrink(x, lambda) = x if |x| > lambda, else 0, over ITERATIONS Dest rows.
 *
 * @tparam APPROXIMATION_MODE: Unused; the kernel is exact in both modes.
 * @tparam ITERATIONS: Number of Dest rows to process; must be even.
 * @param param0: lambda as fp32 bits.
 * @note Call @ref hardshrink_init first: the body advances Dest through ADDR_MOD_6.
 */
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_hardshrink(std::uint32_t param0) {
    static_assert(ITERATIONS % 2 == 0, "calculate_hardshrink() processes dest rows in pairs.");
    // Single comparison against the magnitude: setsgn(v, 1) forms -|v|.
    // param0 contains lambda as FP32 bits. For BF16 inputs, the host pre-rounds
    // lambda to BF16 precision (then re-expands to FP32) so that FP32→FP19b
    // truncation on SFPU preserves the BF16 value exactly. For FP32 inputs,
    // lambda is passed as full FP32, so both input and lambda undergo the same
    // FP32→FP19b truncation, keeping comparisons consistent.
    sfpi::vFloat lambda = Converter::as_float(param0);
    // Two rows per body: the second row's store also advances Dest (see ckernel_sfpu_row_pair.h).
    // The add that forms each difference has a 2-cycle result shadow, and the compiler pads a
    // one-row body with an SFPNOP before the SFPSETCC; with two independent rows it interleaves the
    // second row's setsgn/add into that shadow instead.
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d += 2) {
        sfpi::vFloat v0 = sfpi::dst_reg[0];
        sfpi::vFloat v1 = sfpi::dst_reg[1];
        // Per row, |v| <= lambda is written as d = -|v| + lambda >= 0. Wormhole SFPSETCC only
        // has LT0/GTE0/EQ0/NE0, so the <= form costs two SFPSETCCs plus an SFPCOMPC while this
        // one lands directly on GTE0. Both terms are magnitudes of opposite sign, so the sum cannot
        // overflow and equality still lands on +0.
        //
        // setsgn(v, 1) forms -|v| in one instruction, NaN included, so no separate negation is
        // needed. sfpi::abs is not an option: SFPABS in float mode leaves a -NaN sign-set (see
        // ckernel_sfpu_isclose.h), which would feed a different operand into the compare for -NaN.
        sfpi::vFloat neg_abs_v0 = sfpi::setsgn(v0, 1 /* sgn */);
        sfpi::vFloat neg_abs_v1 = sfpi::setsgn(v1, 1 /* sgn */);
        sfpi::vFloat d0 = neg_abs_v0 + lambda;
        sfpi::vFloat d1 = neg_abs_v1 + lambda;
        v_if(d0 >= 0.0f) { sfpi::dst_reg[0] = 0.0f; }
        v_endif;
        // Row 1's store also advances Dest past both rows, through the ADDR_MOD_6 that hardshrink_init
        // programs (see ckernel_sfpu_row_pair.h).
        v_if(d1 >= 0.0f) { _sfpu_row_pair_store_second_(0.0f); }
        v_endif;
        _sfpu_row_pair_end_();
    }
}

}  // namespace ckernel::sfpu
