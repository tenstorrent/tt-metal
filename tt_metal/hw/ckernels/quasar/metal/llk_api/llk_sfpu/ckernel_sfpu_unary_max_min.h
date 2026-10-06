// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

constexpr std::uint32_t INT32_SIGN_BIT = 0x80000000u;                // sign-magnitude sign bit
constexpr std::uint32_t INT32_SMAG_MIN_MAGNITUDE_MAX = 0x7FFFFFFFu;  // largest sign-magnitude magnitude

/**
 * @brief Re-encode a two's-complement int32 scalar as sign-magnitude, the Dest Int32 encoding.
 *
 * @param value: scalar as a two's-complement int32 bit pattern.
 * @return the same value as a sign-magnitude bit pattern.
 * @note INT32_MIN has no sign-magnitude encoding, so it saturates to -(2^31 - 1).
 */
inline std::uint32_t _unary_max_min_int32_to_smag_(const std::uint32_t value) {
    const std::int32_t v = static_cast<std::int32_t>(value);
    if (v >= 0) {
        return value;
    }
    if (value == INT32_SIGN_BIT) {
        return INT32_SIGN_BIT | INT32_SMAG_MIN_MAGNITUDE_MAX;
    }
    return INT32_SIGN_BIT | static_cast<std::uint32_t>(-v);
}

/**
 * @brief Element-wise max/min of a Dest tile against one uniform scalar: out = max(x, value) or min(x, value).
 *
 * The result is always one of the two operands verbatim — no arithmetic, no rounding — ordered by the SFPU
 * total order (-NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN).
 *
 * @tparam IS_MAX_OP: true selects max, false selects min.
 * @tparam FMT: math-side DataFormat. Int32 takes the sign-magnitude path; every float format takes the fp32
 *         path, so callers may pass Float32 for any float Dest width (the DEFAULT load resolves it).
 * @tparam APPROXIMATION_MODE: accepted for ABI parity but ignored (the select is exact).
 * @tparam ITERATIONS: number of SFP row-pairs per face.
 * @param value: scalar to compare against — an fp32 bit pattern for float FMT, a two's-complement int32 for
 *        DataFormat::Int32.
 * @note Dest Int32 is sign-magnitude while SFPSWAP compares two's complement, so the scalar is re-encoded by
 *       @ref _unary_max_min_int32_to_smag_ and a negative scalar additionally needs the both-negative lanes
 *       corrected. No init call is required.
 */
template <bool IS_MAX_OP, DataFormat FMT, bool APPROXIMATION_MODE, int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_unary_max_min(const std::uint32_t value) {
    static_assert(
        FMT == DataFormat::Float16 || FMT == DataFormat::Float16_b || FMT == DataFormat::Float32 ||
            FMT == DataFormat::Tf32 || FMT == DataFormat::MxFp8R || FMT == DataFormat::MxFp8P ||
            FMT == DataFormat::Int32,
        "Unsupported DataFormat for calculate_unary_max_min().");

    if constexpr (FMT == DataFormat::Int32) {
        const sfpi::vInt s = sfpi::vInt(_unary_max_min_int32_to_smag_(value));
        if (static_cast<std::int32_t>(value) >= 0) {
#pragma GCC unroll 8
            for (int d = 0; d < ITERATIONS; d++) {
                sfpi::vInt x = sfpi::dst_reg[0];
                // INT32 compare is exact on sign-magnitude bits when one operand is non-negative
                x = IS_MAX_OP ? sfpi::max(x, s) : sfpi::min(x, s);
                __builtin_rvtt_sfpnop();  // SFPSWAP -> SFPSTORE spacing
                sfpi::dst_reg[0] = x;
                sfpi::dst_reg++;
            }
        } else {
#pragma GCC unroll 8
            for (int d = 0; d < ITERATIONS; d++) {
                sfpi::vInt x = sfpi::dst_reg[0];
                auto [lo, hi] = sfpi::min_max(x, s);
                __builtin_rvtt_sfpnop();  // SFPSWAP -> consumer spacing
                sfpi::vInt r = IS_MAX_OP ? hi : lo;
                // both operands negative: INT32 compare inverts sign-magnitude order, flip the pick
                v_if(x < 0) { r = IS_MAX_OP ? lo : hi; }
                v_endif;
                sfpi::dst_reg[0] = r;
                sfpi::dst_reg++;
            }
        }
    } else {
        const sfpi::vFloat s = sfpi::as<sfpi::vFloat>(sfpi::vUInt(value));
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat x = sfpi::dst_reg[0];
            x = IS_MAX_OP ? sfpi::max(x, s) : sfpi::min(x, s);  // FP32 total-order compare
            __builtin_rvtt_sfpnop();                            // SFPSWAP -> SFPSTORE spacing
            sfpi::dst_reg[0] = x;
            sfpi::dst_reg++;
        }
    }
}

}  // namespace sfpu
}  // namespace ckernel
