// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_binary_remainder.h"
#include "ckernel_sfpu_recip.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
inline void init_remainder(const uint value, const uint recip) {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpi::vConstFloatPrgm0 = Converter::as_float(value);
    sfpi::vConstFloatPrgm1 = Converter::as_float(recip);
}

// Unary uint32 remainder mirrors the tensor-tensor kernel in ckernel_sfpu_binary_remainder.h.
// The divisor is a compile-time literal, so these runtime checks fold at compile time and only take branches:
//   scalar >= 2^31 -> one conditional subtract
//   power-of-two -> bitmask
//   else (< 2^31) -> range-reduce + full helper (1/|b| hoisted). Unlike Wormhole,Blackhole's helper uses the
//   fractional_mul intrinsic, so divisor width does not change its cost. Small-b branch is not needed.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_remainder_uint32_scalar(uint scalar) {
    sfpi::vInt b = static_cast<int>(scalar);

    if (scalar >= 0x80000000u) {
        // b >= 2^31: a < 2^32 <= 2b and a % b = (a >=u b) ? a - b : a.
        // a is a signed vInt, so a < 0 means the uint32's MSB is set i.e. a >= 2^31.
        // Only a >= 2^31 can be >=u b, so low-half a keeps r = a. For low-half a and high-half b,
        // a - b overflows. Gating on `a < 0` ensures the compare only runs when both operands are in [2^31, 2^32).
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vInt a = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();

            sfpi::vInt r = a;
            v_if(a < 0 && sfpi::vUInt(a) >= sfpi::vUInt(b)) { r = a - b; }
            v_endif;

            sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = r;
            sfpi::dst_reg++;
        }
    } else if ((scalar & (scalar - 1u)) == 0u) {
        // Power of two (non-zero: scalar == 0 is rejected by the host TT_FATAL): a % b == a & (b-1)
        sfpi::vInt mask = static_cast<int>(scalar - 1u);
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vInt a = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
            sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = a & mask;
            sfpi::dst_reg++;
        }
    } else {
        // b < 2^31: range-reduce each a via t = (uint32)a >> 1 (< 2^31). |b| and 1/|b| depend only
        // on the divisor, so compute them once above the loop.
        sfpi::vMag b_mag = sfpi::abs(b);
        sfpi::vFloat inv_b_f = unsigned_remainder_recip(b_mag);

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vInt a = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
            sfpi::vInt t = sfpi::vInt(sfpi::vUInt(a) >> 1);
            sfpi::vInt rt = compute_unsigned_remainder_int32(t, b_mag, inv_b_f);

            // Reload a from DEST instead of keeping it live across the helper
            a = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
            sfpi::vInt x = rt + rt + (a & 1);  // x = 2 * (t % b) + (a & 1), in [0, 2b)

            sfpi::vInt r = x;
            v_if(sfpi::vUInt(x) >= sfpi::vUInt(b)) { r = x - b; }
            v_endif;

            sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = r;
            sfpi::dst_reg++;
        }
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_remainder() {
    // SFPU microcode
    sfpi::vFloat value_tmp = sfpi::vConstFloatPrgm0;
    sfpi::vFloat s = sfpi::abs(value_tmp);
    sfpi::vFloat recip_val = sfpi::abs(sfpi::vConstFloatPrgm1);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = sfpi::dst_reg[0];
        vFloat v = sfpi::abs(val);

        vFloat quotient;
        vInt exp = sfpi::exexp(v * recip_val);
        v_if(exp < 0) { quotient = 0.0f; }
        // Since fp32 has 23 mantissa bits, the LSB represents the fractional part when exp < 23.
        // We effectively round off the fractional bits to zero by right shifting using (exp - 23) and then left
        // shifting it back using (0 - (exp - 23)).
        v_elseif(exp < 23) {
            quotient = sfpi::as<sfpi::vFloat>(
                shft((shft(sfpi::as<sfpi::vUInt>(v * recip_val), (exp - 23))), (0 - (exp - 23))));
        }
        v_else { quotient = v * recip_val; }
        v_endif

        v_if(quotient > v * recip_val) {
            quotient = quotient - 1;
        }
        v_endif;
        v = v - quotient * s;

        v_if(val < 0 && v != 0) { v = s - v; }
        v_endif;

        v_if(value_tmp < 0 && v != 0) { v = v + value_tmp; }
        v_endif;
        v = sfpi::copysgn(v, value_tmp);
        v_if(s == 0) { v = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;

        constexpr auto iter = 10;
        for (int l = 0; l < iter; l++) {
            v_if(v >= s) { v = s - v; }
            v_endif;
        }
        v_if(sfpi::abs(v) - s == 0.0f) { v = 0.0f; }
        v_endif;
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
