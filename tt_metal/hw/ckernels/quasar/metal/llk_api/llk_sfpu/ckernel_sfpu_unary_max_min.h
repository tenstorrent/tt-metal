// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// Sign-magnitude has no encoding for INT32_MIN (0x80000000 is -0 there).
constexpr std::uint32_t UNARY_MAX_MIN_INT32_MIN_BITS = 0x80000000u;

/**
 * @brief out = max(x, value) or min(x, value), one SFPSWAP per row.
 *
 * sfpi >= 7.83 picks the SFPSWAP compare from the vector type: fp32 for vFloat (correct for
 * both-negative pairs), two's-complement int32 for vInt. ckernel_sfpu_gelu.h's note predates this.
 *
 * @tparam FMT: Int32 takes the integer path; any float format takes the fp32 path.
 * @tparam APPROXIMATION_MODE: unused; keeps the dispatcher's (..., APPROX, ITERATIONS) tail.
 * @tparam SIGN_MAGNITUDE_FORMAT: Int32 only; Dest holds sign-magnitude instead of two's complement.
 * @param value: fp32 bits for float FMT, two's-complement int32 for Int32 (not INT32_MIN with SM).
 * @note No SFPNOP after SFPSWAP: SFPSWAP always stalls the next SFPU instruction (TEN-4581).
 */
template <
    bool IS_MAX_OP,
    DataFormat FMT,
    bool APPROXIMATION_MODE,
    int ITERATIONS = SFPU_ITERATIONS,
    bool SIGN_MAGNITUDE_FORMAT = false>
inline void calculate_unary_max_min(const std::uint32_t value) {
    static_assert(
        FMT == DataFormat::Float16 || FMT == DataFormat::Float16_b || FMT == DataFormat::Float32 ||
            FMT == DataFormat::Tf32 || FMT == DataFormat::MxFp8R || FMT == DataFormat::MxFp8P ||
            FMT == DataFormat::Int32,
        "Unsupported DataFormat for calculate_unary_max_min().");
    static_assert(!SIGN_MAGNITUDE_FORMAT || FMT == DataFormat::Int32, "SIGN_MAGNITUDE_FORMAT applies to Int32 only.");

    const auto select = [](auto x, auto s) {
        if constexpr (IS_MAX_OP) {
            return sfpi::max(x, s);
        } else {
            return sfpi::min(x, s);
        }
    };

    if constexpr (FMT == DataFormat::Int32) {
        if constexpr (SIGN_MAGNITUDE_FORMAT) {
            LLK_ASSERT(
                value != UNARY_MAX_MIN_INT32_MIN_BITS,
                "calculate_unary_max_min: INT32_MIN has no sign-magnitude encoding");
        }
        // SM32 wraps the load/store in SFPCAST SM <-> two's complement.
        constexpr sfpi::DataLayout layout = SIGN_MAGNITUDE_FORMAT ? sfpi::DataLayout::SM32 : sfpi::DataLayout::I32;
        const sfpi::vInt s = static_cast<std::int32_t>(value);
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::dst_reg[0].mode<layout>() = select(sfpi::vInt(sfpi::dst_reg[0].mode<layout>()), s);
            sfpi::dst_reg++;
        }
    } else {
        const sfpi::vFloat s = sfpi::as<sfpi::vFloat>(sfpi::vUInt(value));
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::dst_reg[0] = select(sfpi::vFloat(sfpi::dst_reg[0]), s);
            sfpi::dst_reg++;
        }
    }
}

}  // namespace sfpu
}  // namespace ckernel
