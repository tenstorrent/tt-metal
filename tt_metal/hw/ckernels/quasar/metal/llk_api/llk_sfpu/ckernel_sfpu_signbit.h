// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu_comp.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

/// Logical right-shift amount that moves the LREG sign bit down to bit 0, yielding 0/1.
constexpr std::uint32_t SIGNBIT_SHIFT = 31;

#ifndef DISABLE_SFPLOADMACRO
// Sequence 0 is the integer path (shift, store). Sequence 1 is the float path (shift, cast, store).
// @ref init_signbit programs both; @ref calculate_signbit selects one from FMT.
constexpr std::uint32_t SIGNBIT_MACRO_INT = 0;
constexpr std::uint32_t SIGNBIT_MACRO_FLOAT = 1;

// Eight bits per Load Macro slot, high to low: bit 7 retargets VB to VD (else VC=VD), bit 6 writes
// or reads the staging register, bits 5:3 are the issue delay, bits 2:0 are the template (4 + i),
// or 3 for the implicit SFPSTORE back to the load address.
constexpr std::uint32_t SIGNBIT_MACRO_VB_IS_VD = 0x80;
constexpr std::uint32_t SIGNBIT_MACRO_STAGING = 0x40;
constexpr std::uint32_t SIGNBIT_MACRO_SFPSTORE = 3;
#endif

/**
 * @brief Program the dest-increment addr mod used by @ref calculate_signbit.
 *
 * Signbit stores through the same @c ADDR_MOD_6 as the comparison-to-zero family, so it shares
 * that family's init verbatim. Also programs the two SFPLOADMACRO sequences (integer and float);
 * the per-row format stays on the SFPLOADMACRO itself, so this init is not format-templated.
 *
 * @note Call once after @ref _llk_math_eltwise_sfpu_init_ and before @ref calculate_signbit.
 */
inline void init_signbit() {
    init_zero_comp();

#ifndef DISABLE_SFPLOADMACRO
    // Template 4 (round slot): logical >> 31 of VD. SFPSHFT2 mode 6 is the round-unit bitwise
    // shift; modes 0-4 shuffle lanes and are illegal inside a LOADMACRO.
    TTI_SFPSHFT2((-SIGNBIT_SHIFT) & 0xfff, 0, p_sfpu::MACRO_CAPTURE_INSTR4, sfpi::SFPSHFT2_MOD1_SHFT_IMM);
    // Template 5 (simple slot): sign-magnitude int32 -> fp32, exact for 0 and 1.
    TTI_SFPCAST(0, p_sfpu::MACRO_CAPTURE_INSTR5, 0);

    // Sequence 0, integer. Shift result goes to staging; the store reads it one cycle later.
    //   t | Load | Round        | Store
    //   0 | [a]  |              |
    //   1 |      | [a] >> 31    |
    //   2 |      |              | [a]
    {
        constexpr std::uint32_t simple_bits = 0;
        constexpr std::uint32_t round_bits = SIGNBIT_MACRO_VB_IS_VD | SIGNBIT_MACRO_STAGING | (0 << 3) | 4;
        constexpr std::uint32_t store_bits = SIGNBIT_MACRO_STAGING | (1 << 3) | SIGNBIT_MACRO_SFPSTORE;
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, simple_bits);
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, p_sfpconfig::MACRO_SEQ0, 0);
        TTI_SFPNOP(0, 0, 0);  // SFPCONFIG hazard: nothing may issue the cycle after SFPCONFIG
    }

    // Sequence 1, float. The shift stays in VD; the cast consumes it and lands in staging.
    //   t | Load | Simple           | Round     | Store
    //   0 | [a]  |                  |           |
    //   1 |      |                  | [a] >> 31 |
    //   2 |      | cast_fp32([a])   |           |
    //   3 |      |                  |           | [a]
    {
        constexpr std::uint32_t simple_bits = SIGNBIT_MACRO_STAGING | (1 << 3) | 5;
        constexpr std::uint32_t round_bits = SIGNBIT_MACRO_VB_IS_VD | (0 << 3) | 4;
        constexpr std::uint32_t store_bits = SIGNBIT_MACRO_STAGING | (2 << 3) | SIGNBIT_MACRO_SFPSTORE;
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, simple_bits);
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, p_sfpconfig::MACRO_SEQ1, 0);
        TTI_SFPNOP(0, 0, 0);  // SFPCONFIG hazard: nothing may issue the cycle after SFPCONFIG
    }

    // [3:0] StoreMod0 unused. [5:4] both sequences store with the SFPLOADMACRO's format.
    // [9:8] WaitForElapsedInstructions, so a stall cannot let a later macro overtake an earlier one.
    constexpr std::uint32_t macro_ctrl = (0x3 << 4) | (0x3 << 8);
    TTI_SFPCONFIG(macro_ctrl, p_sfpconfig::MACRO_CTRL, 1);
    TTI_SFPNOP(0, 0, 0);  // SFPCONFIG hazard: nothing may issue the cycle after SFPCONFIG
#endif
}

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
 * The SFPLOADMACRO path issues one macro per row (logical >> 31, plus an int32->fp32 cast on the
 * float path) and drains the pipe before returning. @c DISABLE_SFPLOADMACRO keeps the sfpi loop.
 *
 * @tparam APPROXIMATION_MODE: Unused (a bit test is exact); retained for dispatcher signature
 *         symmetry.
 * @tparam FMT: SFPU DataFormat (sfpu_math): Int32/Int16/Int8/UInt16/UInt8, or Float32 for any float
 *         width — the caller must pass Float32 for Float16/Float16_b too, whose DEFAULT sfpmem mode
 *         resolves the actual width from the dest format config. Float16/Float16_b themselves are
 *         rejected: @ref zero_comp_traits only treats Float32 as float, so they would select the
 *         integer result encoding. Anything else is a compile error.
 * @tparam ITERATIONS: Number of SFP-row pairs to process (8 for a 32×16 face).
 * @note Requires @ref init_signbit to have programmed @c ADDR_MOD_6 and the LOADMACRO sequences.
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

#ifndef DISABLE_SFPLOADMACRO
    // Float rows arrive as Float32 and must load DEFAULT: that mode resolves Float16/Float16_b
    // from the dest format config. Integer rows need their own sfpmem mode; DEFAULT would
    // reinterpret them as float. The shifted 0/1 matches in sign-magnitude and two's complement.
    constexpr std::uint32_t seq = traits::is_float ? SIGNBIT_MACRO_FLOAT : SIGNBIT_MACRO_INT;
    constexpr std::uint32_t sfpmem = traits::is_float ? p_sfpu::sfpmem::DEFAULT : _sfpu_sfpmem_type_<FMT>();

    // One macro per row. LREG0-3 rotate so a row still in its shift/cast is not reloaded.
    // Unrolling constant-folds each SFPLOADMACRO encoding. The trailing NOPs are one fewer than
    // the sequence length (3 float, 2 integer), so the last store retires before return.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOADMACRO(seq, d & 3, sfpmem, ADDR_MOD_6, 0, 0, 0);
    }
    TTI_SFPNOP(0, 0, 0);
    TTI_SFPNOP(0, 0, 0);
    if constexpr (traits::is_float) {
        TTI_SFPNOP(0, 0, 0);
    }
#else
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
#endif
}

}  // namespace sfpu
}  // namespace ckernel
