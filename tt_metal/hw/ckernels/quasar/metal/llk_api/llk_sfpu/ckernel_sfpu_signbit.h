// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_defs.h"
#include "ckernel_sfpu_comp.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

/// Logical right-shift amount that moves the LREG sign bit down to bit 0, yielding 0/1.
constexpr std::uint32_t SIGNBIT_SHIFT = 31;

/**
 * @brief Program the dest-increment addr mod used by @ref calculate_signbit.
 *
 * Signbit stores through the same @c ADDR_MOD_6 as the comparison-to-zero family, so it shares
 * that family's init verbatim.
 *
 * @note Call once after @ref _llk_math_eltwise_sfpu_init_ and before @ref calculate_signbit.
 */
inline void init_signbit() { init_zero_comp(); }

/**
 * @brief Element-wise sign-bit test over a tile, written as 1/0 in FMT's native encoding.
 *
 * A bit test, not a compare: the result is bit 31 of the element's LREG image, so @c -0.0 and a
 * negative NaN yield 1 where an IEEE @c <0 compare would yield 0. (Quasar's less_than_zero, see
 * @ref _zero_comp_pred_, is itself a bit-31 test guarded by @c mag!=0, so on hardware signbit differs
 * from it only at @c -0.0.) Load, store and result encoding come
 * from @ref zero_comp_traits; the store rides @c ADDR_MOD_6 (dest.incr=2) to advance the dest
 * counter, so the loop needs no dst_reg++.
 *
 * @tparam APPROXIMATION_MODE: Unused (a bit test is exact); retained for dispatcher signature
 *         symmetry.
 * @tparam FMT: SFPU DataFormat (sfpu_math): Int32/Int16/Int8/UInt16/UInt8, or Float32 for any float
 *         width — the caller must pass Float32 for Float16/Float16_b too, whose DEFAULT sfpmem mode
 *         resolves the actual width from the dest format config. Float16/Float16_b themselves are
 *         rejected: @ref zero_comp_traits only treats Float32 as float, so they would select the
 *         integer result encoding. Anything else is a compile error.
 * @tparam ITERATIONS: Number of SFP-row pairs to process (8 for a 32×16 face).
 * @note Requires @ref init_signbit to have programmed @c ADDR_MOD_6.
 */
template <bool APPROXIMATION_MODE, DataFormat FMT, int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_signbit() {
    constexpr bool is_int_fmt = FMT == DataFormat::Int32 || FMT == DataFormat::Int16 || FMT == DataFormat::Int8 ||
                                FMT == DataFormat::UInt16 || FMT == DataFormat::UInt8;
    // Every float width must arrive canonicalized to Float32: zero_comp_traits keys is_float on
    // Float32 alone, so Float16/Float16_b would silently take the integer result encoding.
    constexpr bool is_float_fmt = FMT == DataFormat::Float32;
    static_assert(
        is_int_fmt || is_float_fmt,
        "calculate_signbit: unsupported FMT (expected an integer format, or Float32 for any float width)");

    using traits = zero_comp_traits<FMT>;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vInt bits = traits::load();                              // sign sits at LREG bit 31 for every format
        sfpi::vUInt s = sfpi::as<sfpi::vUInt>(bits) >> SIGNBIT_SHIFT;  // logical shift -> 0/1
        if constexpr (traits::is_float) {
            traits::store(
                sfpi::convert<sfpi::vFloat>(sfpi::as<sfpi::vSMag>(s), sfpi::RoundMode::Nearest));  // 0/1 -> 0.0f/1.0f
        } else {
            traits::store(sfpi::as<sfpi::vInt>(s));  // 0/1 identical in SM and 2's complement
        }
    }
}

}  // namespace sfpu
}  // namespace ckernel
