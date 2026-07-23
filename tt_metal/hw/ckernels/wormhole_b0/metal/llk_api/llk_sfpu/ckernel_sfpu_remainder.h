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

// Exact unsigned remainder for operands known to be in [0, 2^31) with b < 2^11. The divisor is a
// compile-time literal, so b < 2^11 is guaranteed for every lane. The mid/high chunk multiplies
// (q1*b1, correction*b1, correction*b2) of the full helper can be dropped statically. 1/b and low 11-bit
// chunk b0 are precomputed by the caller. Abs(a) is reloaded after q*b to reduce SFPU register pressure.
sfpi_inline sfpi::vInt compute_unsigned_remainder_small_b(
    const sfpi::vInt& a_unsigned, const sfpi::vInt& b_unsigned, const sfpi::vFloat& inv_b_f, const sfpi::vFloat& b0) {
    sfpi::vMag a = sfpi::abs(a_unsigned);
    sfpi::vFloat a_f = sfpi::convert<sfpi::vFloat>(a, sfpi::RoundMode::Nearest);

    sfpi::vFloat q_f = a_f * inv_b_f + sfpi::vConstFloatPrgm0;
    sfpi::vMag q = sfpi::exman(q_f);

    constexpr float MANTISSA_ALIGNMENT_OFFSET = 8388608.0f;  // 2^23
    sfpi::vMag MASK_11{0x7ff};
    sfpi::vFloat q1 = sfpi::convert<sfpi::vFloat>(q & MASK_11, sfpi::RoundMode::Nearest);
    sfpi::vFloat q2 = sfpi::convert<sfpi::vFloat>(q >> 11, sfpi::RoundMode::Nearest);

    sfpi::vFloat lo = q1 * b0 + MANTISSA_ALIGNMENT_OFFSET;
    sfpi::vFloat hi = q2 * b0 + MANTISSA_ALIGNMENT_OFFSET;

    sfpi::vUInt qb = sfpi::exman(lo) << 11;
    qb += sfpi::exman(hi) << 22;

    a = sfpi::abs(a_unsigned);
    sfpi::vInt r{a - qb};

    sfpi::vFloat r_f = sfpi::convert<sfpi::vFloat>(sfpi::abs(r), sfpi::RoundMode::Nearest);
    sfpi::vFloat correction_f = r_f * inv_b_f;
    auto correction = sfpi::convert<sfpi::vUInt16>(correction_f, sfpi::RoundMode::Nearest);
    correction_f = sfpi::convert<sfpi::vFloat>(correction, sfpi::RoundMode::Nearest);

    sfpi::vFloat low = correction_f * b0 + MANTISSA_ALIGNMENT_OFFSET;
    sfpi::vInt tmp{sfpi::exman(low)};

    v_if(r < 0) { tmp = -tmp; }
    v_endif;
    r -= tmp;

    sfpi::vMag b = sfpi::abs(b_unsigned);
    v_if(r < 0 && (r - 1) < 0) { r += b; }
    v_elseif(r >= b) { r -= b; }
    v_endif;

    return r;
}

// Unary uint32 remainder mirrors the tensor-tensor kernel in ckernel_sfpu_binary_remainder.h.
// The divisor is a compile-time literal, so these runtime checks fold at compile time and only take branches:
//   scalar >= 2^31 -> one conditional subtract
//   power-of-two -> bitmask
//   scalar < 2^11 -> range-reduce + small-b helper (b0 + 1/b hoisted, skips high chunks)
//   else (< 2^31) -> range-reduce + full helper (1/b hoisted)
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
    } else if (scalar < (1u << 11)) {
        // b < 2^11: range-reduce each a via t = (uint32)a >> 1 (< 2^31). 1/b and the single 11-bit
        // chunk b0 of the divisor are loop-invariant, so hoist them. The small-b helper skips the mid/high
        // chunk multiplies that the full helper would do.
        sfpi::vFloat inv_b_f = unsigned_remainder_recip(b);
        sfpi::vMag b_mag = sfpi::abs(b);
        sfpi::vMag MASK_11{0x7ff};
        sfpi::vFloat b0 = sfpi::convert<sfpi::vFloat>(b_mag & MASK_11, sfpi::RoundMode::Nearest);

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vInt a = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
            sfpi::vInt t = sfpi::vInt(sfpi::vUInt(a) >> 1);
            sfpi::vInt rt = compute_unsigned_remainder_small_b(t, b, inv_b_f, b0);

            // Reload a from DEST instead of keeping it live across the helper
            a = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
            sfpi::vInt x = rt + rt + (a & 1);  // x = 2 * (t % b) + (a & 1), in [0, 2b)

            // x % b = (x >=u b) ? x - b : x
            sfpi::vInt r = x;
            v_if(sfpi::vUInt(x) >= sfpi::vUInt(b)) { r = x - b; }
            v_endif;

            sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = r;
            sfpi::dst_reg++;
        }
    } else {
        // b in [2^11, 2^31): range-reduce each a via t = (uint32)a >> 1 (< 2^31) so the helper
        // always sees operands in [0, 2^31). Hoist only 1/b. The full helper re-extracts b's chunks
        // under its own register-pressure schedule (hoisting b1/b2 here spills on Wormhole).
        sfpi::vFloat inv_b_f = unsigned_remainder_recip(b);

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vInt a = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
            sfpi::vInt t = sfpi::vInt(sfpi::vUInt(a) >> 1);
            sfpi::vInt rt = compute_unsigned_remainder_int32(t, b, inv_b_f);

            // Reload a from DEST instead of keeping it live across the helper
            a = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
            sfpi::vInt x = rt + rt + (a & 1);  // x = 2 * (t % b) + (a & 1), in [0, 2b)

            // x % b = (x >=u b) ? x - b : x
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
    sfpi::vFloat value_tmp = vConstFloatPrgm0;
    sfpi::vFloat s = sfpi::abs(value_tmp);
    sfpi::vFloat recip_val = sfpi::abs(vConstFloatPrgm1);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = dst_reg[0];
        sfpi::vFloat v = sfpi::abs(val);

        sfpi::vFloat quotient;
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
