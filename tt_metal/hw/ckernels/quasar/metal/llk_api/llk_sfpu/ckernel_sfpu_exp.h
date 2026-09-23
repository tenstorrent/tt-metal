// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>
#include <type_traits>
#include <cstdint>

#include "ckernel.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu_srcs.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_operand.h"

namespace ckernel {
namespace sfpu {

// Round-to-nearest-even of a float to its integer value, returning both the rounded float (result)
// and the integer (k_int). Uses the Hacker's Delight 2^23 + 2^22 trick: adding that constant forces
// the fractional bits out, and differencing the raw bit patterns recovers the integer. Only uses
// add/sub plus a bit reinterpret, so it is portable to Quasar (unlike the sign-magnitude round the
// Blackhole kernel interleaves). Valid for |z| < 2^22, which covers exp's reduced argument.
sfpi_inline sfpi::vFloat _sfpu_round_to_nearest_int32_(sfpi::vFloat z, sfpi::vInt& k_int) {
    const sfpi::vFloat c231 = 12582912.0f;  // 2^23 + 2^22
    sfpi::vFloat tmp = z + c231;
    k_int = sfpi::as<sfpi::vInt>(tmp) - sfpi::as<sfpi::vInt>(c231);
    return tmp - c231;
}

/*
 * Branch-free float->int32 conversion for the 21f exp construction, taken from the Blackhole kernel of
 * the same name. Requires 0 <= val < 128.0f and assumes val was already divided by 2^23, so the result
 * is scaled by 2^23 (otherwise the shift would have to be exp - 23).
 *
 * Safe on Quasar: the exponent is non-negative over that range, so the shift amount reads the same in
 * sign-magnitude and two's complement.
 */
sfpi_inline sfpi::vInt _float_to_int32_for_exp_21f_(sfpi::vFloat val) {
    sfpi::vInt exp = sfpi::exexp(val);
    sfpi::vInt man =
        sfpi::exman(val, sfpi::MantissaMode::ImplicitOne);  // get mantissa with implicit bit (man in [1; 2])
    man = sfpi::shft(man, exp, sfpi::ShiftMode::Logical);
    return man;
}

/*
 * The _sfpu_exp_fp32_accurate_ code is derived from code by Norbert Juffa.
 *
 * Copyright (c) 2015-2021, Norbert Juffa
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
// Non-finite behaviour of this path:
//   +NaN -> NaN    -NaN -> 0    +Inf -> +Inf    -Inf -> 0
// -NaN diverges from Blackhole, which returns NaN for either sign: a negative-signed NaN drives i
// negative, so the lane lands in the underflow arm. Derived from simulation of this routine, NOT
// silicon-verified -- no Quasar target available.
sfpi_inline sfpi::vFloat _sfpu_exp_fp32_accurate_(sfpi::vFloat a) {
    sfpi::vInt i;
    sfpi::vFloat f, r, j;

    // j = round(a / ln2) (as a float) and i = the same value as an integer, interleaved with the
    // first coefficient of the polynomial.
    r = 1.37805939e-3f;
    j = _sfpu_round_to_nearest_int32_(1.442695f * a, i);

    // f = a - j*ln2 (two-part Cody-Waite).
    f = j * -6.93145752e-1f + a;
    f = j * -1.42860677e-6f + f;

    // r = exp(f) on [-ln2/2, ln2/2] via a degree-6 minimax polynomial in Horner form.
    r = r * f + 8.37312452e-3f;  // 0x1.125edcp-7
    r = r * f + 4.16695364e-2f;  // 0x1.555b5ap-5
    r = r * f + 1.66664720e-1f;  // 0x1.555450p-3
    r = r * f + 4.99999851e-1f;  // 0x1.fffff6p-2
    r = r * f + 1.0f;
    r = r * f + 1.0f;

    // exp(a) = 2^i * exp(f), applied via the result's biased exponent e. The legal IEEE-754 range
    // is 1..254, and i pushes e outside it for |a| beyond ~88. Writing (i + 127) << 23 directly
    // would wrap into the sign bit there -- Quasar's integer adder is 32-bit two's complement and
    // SHFT discards the high bits (Quasar/Trinity SFPU MAS), so a = -100 yields i + 127 = -17 ->
    // 0xF7800000 = -2^112 rather than ~0. Nothing routes large-magnitude inputs away from this
    // path (calculate_exponential selects purely on EN_32BIT_DEST / APPROXIMATION_MODE).
    //
    // Seed y with the overflow result unpredicated, as the Blackhole source does, and take the
    // exponent path only when e is in range. Seeding from a (not from the polynomial, which is
    // NaN for a = +-Inf because f = j*-ln2 + a is -Inf + Inf) keeps exp(+Inf) = +Inf.
    //
    // NB: the earlier Quasar port bug was elsewhere -- the Blackhole kernel's sign-magnitude
    // rounding (abs(as<vInt>(convert<vSMag16>(x))) + copysgn feeding a two's-complement add),
    // which relies on Blackhole's integer-format behaviour. _sfpu_round_to_nearest_int32_ above
    // replaces that.
    sfpi::vFloat y = a * std::numeric_limits<float>::infinity();
    sfpi::vInt e = sfpi::exexp(r, sfpi::ExponentMode::Biased) + i;
    v_if(e < 255) {
        y = sfpi::setexp(r, e);
        v_if(e < 1) { y = 0.0f; }  // underflow, incl. subnormals
        v_endif;
    }
    v_endif;
    return y;
}

// EXP algorithm policies: apply(vFloat) -> vFloat. The operand loop is independent of the choice.

// HW nonlinear LUT (SFPNONLINEAR EXP); ~1 ULP once the result lands in a bf16 register.
struct ExpHwLut {
    sfpi_inline static sfpi::vFloat apply(sfpi::vFloat x) { return sfpi::approx_exp(x); }
};

// fp32 Cody-Waite + degree-6 minimax polynomial (ported from Blackhole); for 32-bit results only.
struct ExpFp32Accurate {
    sfpi_inline static sfpi::vFloat apply(sfpi::vFloat x) { return _sfpu_exp_fp32_accurate_(x); }
};

// Accurate polynomial only for a 32-bit result in non-approximate mode; LUT otherwise.
template <bool APPROXIMATION_MODE, bool FP32_RESULT>
using ExpAlgo = std::conditional_t<(!FP32_RESULT || APPROXIMATION_MODE), ExpHwLut, ExpFp32Accurate>;

/**
 * @brief EXP on independently located floating-point operands: output = Algo::apply(input).
 *
 * Advances explicit indices only; the caller owns setup and synchronization. Input/output
 * ranges must coincide or be disjoint.
 *
 * @tparam Algo: @ref ExpHwLut or @ref ExpFp32Accurate (see @ref ExpAlgo).
 */
template <class Algo, int ITERATIONS, class Input, class Output>
sfpi_inline void calculate_exponential_operands(const Input& input, const Output& output) {
    static_assert(ITERATIONS > 0, "EXP requires at least one SFPI access");
    static_assert(
        std::is_same_v<typename Input::value_type, sfpi::vFloat> &&
            std::is_same_v<typename Output::value_type, sfpi::vFloat>,
        "EXP requires floating-point operands");
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        output.store(d, Algo::apply(input.load(d)));
    }
}

// Calculates EXP over a Dest span (one face by default). Quasar exposes two implementations:
//   - approximate exp via the HW nonlinear lookup table (sfpi::approx_exp), and
//   - full-precision fp32 exp (_sfpu_exp_fp32_accurate_, ported from Blackhole).
// The LUT is ~1 ULP once the result lands in a bf16 Dest, so the accurate path is only worth
// running for a 32-bit Dest in non-approximate mode; every bf16 case (and any explicit approx
// request) uses the LUT. EN_32BIT_DEST (is_fp32_dest_acc_en) selects the accurate path.
template <
    bool APPROXIMATION_MODE,
    bool EN_32BIT_DEST,
    bool SCALE_EN /*maybe_unused*/ = false,
    int ITERATIONS = SFPU_ITERATIONS,
    bool CLAMP_NEGATIVE /*maybe_unused*/ = true>
void calculate_exponential([[maybe_unused]] const std::uint32_t exp_base_scale_factor = p_sfpu::kCONST_1_FP16B) {
    static_assert(SCALE_EN == false, "Non-default SCALE_EN not supported in Quasar exp");
    static_assert(CLAMP_NEGATIVE == true, "Non-default CLAMP_NEGATIVE not supported in Quasar exp");
    LLK_ASSERT(
        exp_base_scale_factor == p_sfpu::kCONST_1_FP16B,
        "Scaling is not supported in the current version of exp on Quasar.");
    using Operand = SfpuOperand<SfpuReg::Dest, SfpiFormat<sfpi::DataLayout::Default, sfpi::vFloat>>;
    using Algo = ExpAlgo<APPROXIMATION_MODE, EN_32BIT_DEST>;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_exponential_operands<Algo, 1>(Operand{}, Operand{});
        sfpi::dst_reg++;
    }
}

// Reference adapter for the unified Dest/SrcS API; not yet wired into a test.
/**
 * @brief EXP over one SrcS slice (slots per @ref SrcsLayout), algorithm per @ref ExpAlgo.
 *
 * @tparam LAYOUT: Load and store layout, values = <F16a/F16b/F32>; unpack destination and pack
 *         source formats must match.
 * @note The caller runs unpack/pack and clears the SrcS valids after this call, as
 *       llk_sfpu_srcs_unary does.
 */
template <bool APPROXIMATION_MODE, sfpi::DataLayout LAYOUT>
sfpi_inline void calculate_exponential_srcs() {
    using Layout = SrcsLayout<LAYOUT>;
    using Operand = SfpuOperand<SfpuReg::SrcS, SfpiFormat<LAYOUT, sfpi::vFloat>>;
    using Algo = ExpAlgo<APPROXIMATION_MODE, LAYOUT == sfpi::DataLayout::F32>;
    calculate_exponential_operands<Algo, Layout::ops>(Operand{Layout::in0}, Operand{Layout::out});
}

template <
    bool APPROXIMATION_MODE /*maybe_unused*/,
    std::uint32_t scale /*maybe_unused*/ = 0x3F800000,
    bool CLAMP_NEGATIVE /*maybe_unused*/ = true,
    bool EN_32BIT_DEST /*maybe_unused*/>
void exp_init() {
    static_assert(scale == 0x3F800000, "Non-default scale not supported in Quasar exp");
    static_assert(CLAMP_NEGATIVE == true, "Non-default CLAMP_NEGATIVE not supported in Quasar exp");
    llk_math_eltwise_unary_sfpu_init<SfpuType::exponential>();
}

}  // namespace sfpu
}  // namespace ckernel
