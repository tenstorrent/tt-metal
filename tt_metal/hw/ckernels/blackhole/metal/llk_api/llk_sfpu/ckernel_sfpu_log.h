// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

/*
 * The log(x) code is derived from code by Norbert Juffa.
 *
 * Copyright (c) 2015-2023, Norbert Juffa
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "ckernel_sfpu_polyval.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// Biased fp32 exponent of 1.0: setexp(x, FP32_EXP_BIAS) rescales x into [1, 2).
constexpr std::uint32_t FP32_EXP_BIAS = 127;

/**
 * @brief Constants of the ln(x) cubic used by _calculate_log_body_ / _init_log_.
 *
 * Minimax cubic for ln(x) on [1, 2), x * (x * (x * A + B) + C) + D, max |error| 4.4e-04.
 * _init_log_ holds LN2, A and B in vConstFloatPrgm0-2. C and D are bound by the caller to a
 * vFloat outside its row loop so they stay in an LREG: sfpi 7.83.0 reloads a non-16-bit-exact
 * fp32 literal with an SFPLOADI pair at every use.
 */
struct LogPoly {
    static constexpr float LN2 = 0.69314718f;
    static constexpr float A = 0.10968964f;
    static constexpr float B = -0.72910421f;
    static constexpr float C = 2.11263230f;
    static constexpr float D = -1.49277612f;
};

/**
 * @brief Constants of the ln(x) cubic used by _calculate_log_body_no_init_, which owns no
 * program constant register.
 *
 * x * (x * (x * A - B) + C) - D, rminimax over [1, 2]; B and D are subtracted so the SFPMAD
 * sequence stays fixed. LN2 is deliberately the coarse 0.692871: lgamma, digamma and POW
 * results depend on it.
 */
struct LogPolyNoInit {
    static constexpr float LN2 = 0.692871f;
    static constexpr float A = 0x2.44734p-4f;
    static constexpr float B = 0xd.e712ap-4f;
    static constexpr float C = 0x2.4f5388p+0f;
    static constexpr float D = 0x1.952992p+0f;
};

/**
 * @brief ln(in) from the cubic on the mantissa plus exponent * ln2, before the ln(0) = -inf fixup.
 *
 * @param in: Input value.
 * @param c: LogPoly::C, bound by the caller outside its row loop.
 * @param d: LogPoly::D, bound by the caller outside its row loop.
 * @note Call @ref _init_log_ before this function; it reads vConstFloatPrgm0-2.
 */
sfpi_inline sfpi::vFloat _calculate_log_series_(const sfpi::vFloat in, const sfpi::vFloat c, const sfpi::vFloat d) {
    ////////////////////////////
    // "normalize to calculation range"
    ////////////////////////////
    sfpi::vFloat x = setexp(in, FP32_EXP_BIAS);  // rescale in to [1, 2)

    ////////////////////////////
    // Minimax cubic approximation of ln(x) on x in [1, 2), Horner form:
    // x * (x * (x * A + B) + C) + D, see LogPoly
    ////////////////////////////
    sfpi::vFloat a = sfpi::vConstFloatPrgm1;
    sfpi::vFloat b = sfpi::vConstFloatPrgm2;
    sfpi::vFloat series_result = x * (x * (x * a + b) + c) + d;

    ////////////////////////////
    // Convert exponent to float
    ////////////////////////////
    auto exp = sfpi::convert<sfpi::vSMag>(sfpi::exexp(in));

    sfpi::vFloat expf = sfpi::convert<sfpi::vFloat>(exp, sfpi::RoundMode::Nearest);
    sfpi::vFloat vConstLn2 = sfpi::vConstFloatPrgm0;
    return expf * vConstLn2 + series_result;  // exp correction: ln(1+x) + exp*ln(2)
}

/**
 * @brief One row of ln(x) in place in Dest.
 *
 * The zero test is bitwise, so -0.0 is not mapped to -inf and _calculate_log_, the only
 * caller, returns a finite value for it. A caller that holds its operand in a register uses
 * @ref _calculate_log_body_on_reg_ instead, whose biased-exponent test maps +-0 and the
 * denormals to -inf.
 *
 * @param c: LogPoly::C, bound outside the caller's row loop.
 * @param d: LogPoly::D, bound outside the caller's row loop.
 * @param dst_idx: Dest tile index of the row.
 * @note Call @ref _init_log_ first.
 */
sfpi_inline void _calculate_log_body_(const sfpi::vFloat c, const sfpi::vFloat d, const std::uint32_t dst_idx = 0) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr std::uint32_t dst_tile_size_sfpi = 32;

    sfpi::vFloat in = sfpi::dst_reg[dst_idx * dst_tile_size_sfpi];
    sfpi::vFloat result = _calculate_log_series_(in, c, d);

    ////////////////////////////
    // Base case when input is 0. ln(0) = -inf (bitwise test, -0.0 is not selected; see above)
    ////////////////////////////
    v_if(in == 0.0F) { result = -std::numeric_limits<float>::infinity(); }
    v_endif;

    sfpi::dst_reg[dst_idx * dst_tile_size_sfpi] = result;
}

/**
 * @brief One row of log_base(x) = ln(x) * base_scale in place in Dest.
 *
 * The ln(0) = -inf fixup is applied after the scaling. Same bitwise zero test as
 * _calculate_log_body_, so -0.0 is not mapped to -inf.
 *
 * @param c: LogPoly::C, bound outside the caller's row loop.
 * @param d: LogPoly::D, bound outside the caller's row loop.
 * @param base_scale: 1/ln(base), bound outside the caller's row loop.
 * @param dst_idx: Dest tile index of the row.
 * @note Call @ref _init_log_ first.
 */
sfpi_inline void _calculate_log_with_base_body_(
    const sfpi::vFloat c, const sfpi::vFloat d, const sfpi::vFloat base_scale, const std::uint32_t dst_idx = 0) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr std::uint32_t dst_tile_size_sfpi = 32;

    sfpi::vFloat in = sfpi::dst_reg[dst_idx * dst_tile_size_sfpi];
    sfpi::vFloat result = _calculate_log_series_(in, c, d) * base_scale;

    ////////////////////////////
    // Base case when input is 0. ln(0) = -inf (bitwise test, -0.0 is not selected)
    ////////////////////////////
    v_if(in == 0.0F) { result = -std::numeric_limits<float>::infinity(); }
    v_endif;

    sfpi::dst_reg[dst_idx * dst_tile_size_sfpi] = result;
}

/**
 * @brief ln(in) of a value already held in a register, with no Dest load or store.
 *
 * Same arithmetic as _calculate_log_body_, for a caller that holds its operand in a register
 * and would otherwise store it to Dest, run the in-place body and reload the result (two
 * SFPSTOREs and two SFPLOADs per row).
 *
 * The -inf lane is every input whose biased exponent is 0: +-0 *and* the denormals. That is
 * what the Dest round trip it replaces produces: SFPSTORE flushes a denormal to zero (measured
 * on Blackhole, fp32 and bf16 Dest), so the in-place body only ever sees 0 there. An
 * `in == 0.0F` test would instead return a finite value for a denormal (about -88.0..-87.3
 * for a positive one: setexp gives 1.m and exexp the raw 0 - 127, since SFPEXEXP does not
 * normalize).
 *
 * @param in: Input value.
 * @param c: LogPoly::C, bound outside the caller's row loop.
 * @param d: LogPoly::D, bound outside the caller's row loop.
 * @note Call @ref _init_log_ first.
 */
sfpi_inline sfpi::vFloat _calculate_log_body_on_reg_(
    const sfpi::vFloat in, const sfpi::vFloat c, const sfpi::vFloat d) {
    sfpi::vFloat result = _calculate_log_series_(in, c, d);

    v_if(sfpi::exexp(in, sfpi::ExponentMode::Biased) == 0) { result = -std::numeric_limits<float>::infinity(); }
    v_endif;

    return result;
}

/**
 * @brief ln(base) without any program constant register.
 *
 * ln2 and D come from the caller: a loop with two spare LREGs (lgamma) keeps them resident, a
 * loop with none (digamma) passes the plain floats and they are materialised at use. A, B and
 * C are always materialised at use. The zero test is bitwise; no caller delivers -0.0.
 *
 * @tparam Ln2T: sfpi::vFloat (LREG-resident) or float (materialised at use).
 * @tparam DT: sfpi::vFloat (LREG-resident) or float (materialised at use).
 * @param base: Input value.
 * @param ln2: LogPolyNoInit::LN2.
 * @param d: LogPolyNoInit::D (subtracted).
 */
template <typename Ln2T, typename DT>
sfpi_inline sfpi::vFloat _calculate_log_body_no_init_(sfpi::vFloat base, const Ln2T ln2, const DT d) {
    // Normalize base to calculation range
    sfpi::vFloat x = setexp(base, FP32_EXP_BIAS);  // rescale base to [1, 2)

    // 3rd order polynomial approx - determined using rminimax over [1,2], see LogPolyNoInit
    sfpi::vFloat series_result = x * (x * (x * LogPolyNoInit::A - LogPolyNoInit::B) + LogPolyNoInit::C) - d;

    // Convert exponent to float
    auto exp = sfpi::convert<sfpi::vSMag>(sfpi::exexp(base));
    sfpi::vFloat expf = sfpi::convert<sfpi::vFloat>(exp, sfpi::RoundMode::Nearest);

    // De-normalize to original range
    sfpi::vFloat log_result = expf * ln2 + series_result;  // exp correction: ln(1+x) + exp*ln(2)

    // Base case when input is 0. ln(0) = -inf (bitwise test, -0.0 is not selected)
    v_if(base == 0.0f) { log_result = -std::numeric_limits<float>::infinity(); }
    v_endif;

    return log_result;
}

/**
 * @brief ln(base) with every constant materialised at its use, for a caller whose loop has no
 * LREG to spare.
 *
 * @param base: Input value.
 */
sfpi_inline sfpi::vFloat _calculate_log_body_no_init_(sfpi::vFloat base) {
    return _calculate_log_body_no_init_(base, LogPolyNoInit::LN2, LogPolyNoInit::D);
}

/**
 * @brief ln(x), or log_base(x) when HAS_BASE_SCALING, on iterations rows of Dest in place.
 *
 * C, D and the base scale are bound once here and stay in LREGs across the row loop.
 * -0.0 is not mapped to -inf (bitwise zero test).
 *
 * @tparam APPROXIMATION_MODE: Unused.
 * @tparam HAS_BASE_SCALING: Multiply ln(x) by the base scale, values = <true/false>
 * @tparam ITERATIONS: Unused; the row count is the runtime iterations argument.
 * @param iterations: Number of Dest rows to process.
 * @param log_base_scale_factor: 1/ln(base) as fp16a bits; read only when HAS_BASE_SCALING.
 * @note Call @ref _init_log_ first.
 */
template <bool APPROXIMATION_MODE, bool HAS_BASE_SCALING, int ITERATIONS>
inline void _calculate_log_(const int iterations, std::uint32_t log_base_scale_factor) {
    const sfpi::vFloat c = LogPoly::C;
    const sfpi::vFloat d = LogPoly::D;
    if constexpr (HAS_BASE_SCALING) {
        const sfpi::vFloat base_scale = sfpi::sFloat16a(log_base_scale_factor);
#pragma GCC unroll 8
        for (int i = 0; i < iterations; i++) {
            _calculate_log_with_base_body_(c, d, base_scale);
            sfpi::dst_reg++;
        }
    } else {
#pragma GCC unroll 8
        for (int i = 0; i < iterations; i++) {
            _calculate_log_body_(c, d);
            sfpi::dst_reg++;
        }
    }
}

/**
 * @brief Program vConstFloatPrgm0-2 with LogPoly::LN2, A and B for _calculate_log_body_.
 *
 * @tparam APPROXIMATION_MODE: Unused; the same constants serve both modes.
 */
template <bool APPROXIMATION_MODE>
inline void _init_log_() {
    sfpi::vConstFloatPrgm0 = LogPoly::LN2;
    sfpi::vConstFloatPrgm1 = LogPoly::A;
    sfpi::vConstFloatPrgm2 = LogPoly::B;
}

template <bool FAST_APPROX, bool HAS_BASE_SCALING, bool is_fp32_dest_acc_en, bool IS_BASE_TWO = false>
sfpi_inline sfpi::vFloat calculate_log_body(sfpi::vFloat a, const uint log_base_scale_factor) {
    if constexpr (!FAST_APPROX) {
        // normalise a (-0.0 and subnormals become +0.0). This must run before
        // the exponent read below so that -0.0 follows the same path as +0.0
        // and log(±0.0) returns -inf, matching torch.
        a = a * 1.0f + 0.0f;
    }

    sfpi::vFloat three_quarters = 0.75f;
    sfpi::vInt e = sfpi::as<sfpi::vInt>(a) - sfpi::as<sfpi::vInt>(three_quarters);
    e = sfpi::as<sfpi::vInt>(sfpi::setman(sfpi::as<sfpi::vFloat>(e), 0));
    sfpi::vFloat m = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(a) - e);
    sfpi::vFloat result = std::numeric_limits<float>::quiet_NaN();

    // m in [0.75, 1.5). Compute log1p(m - 1) for m - 1 in [-0.25, 0.5).
    m -= 1.0f;

    v_if(a >= 0.0f) {
        sfpi::vFloat r;
        sfpi::vFloat s = m * m;
        sfpi::vFloat e_float;
        if constexpr (is_fp32_dest_acc_en) {
            r = -0x1.92cp-5f;
            r = r * m + 0x1.b84p-4f;
            r = r * m + -0x1.0c4p-3f;
            r = r * m + 0x1.274p-3f;
            r = r * m + -0x1.55p-3f;
            r = r * m + 0x1.998p-3f;
            sfpi::vMag abs_e = sfpi::abs(e);
            r = r * m + sfpi::vConstFloatPrgm1;
            e_float = sfpi::convert<sfpi::vFloat>(abs_e, sfpi::RoundMode::Nearest);
            r = r * m + sfpi::vConstFloatPrgm2;
            sfpi::vFloat neg_half = -0.5f;
            r = __builtin_rvtt_sfpmad(r.get(), m.get(), neg_half.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        } else {
            sfpi::vMag abs_e = sfpi::abs(e);
            sfpi::vFloat neg_quarter = -0.25f;
            r = neg_quarter * m + sfpi::vConstFloatPrgm1;
            e_float = sfpi::convert<sfpi::vFloat>(abs_e, sfpi::RoundMode::Nearest);
            r = r * m + sfpi::vConstFloatPrgm2;
        }

        // Handle special cases:
        //
        //   input 0.0  -> -inf
        //   input +inf -> +inf
        //   input NaN  -> NaN
        //
        // In the non-fast path, earlier normalisation maps -0.0 and subnormals
        // to +0.0. addexp(a, -1) wraps exponent 0 to 255, so zero becomes
        // +inf; exponent 255 values (Inf/NaN) are left unchanged.
        a = sfpi::addexp(a, -1);

        r = r * s + m;
        e_float = sfpi::copysgn(e_float, sfpi::as<sfpi::vFloat>(e));
        if constexpr (IS_BASE_TWO) {
            // log2 takes an exact path.  Scaling the finished natural-log sum, as
            // result *= 1/ln(2), also scales the exponent contribution by
            // ln(2) * (1/ln(2)), which does not round to exactly 1 in float, so log2 of
            // an exact power of two came back an ULP low for 46 of the 254 representable
            // exponents.
            //
            // Applying the base change to the mantissa term only leaves the exponent
            // term as e_float * 2^-23.  The exponent arrives here as k << 23 and 2^-23 is
            // a power of two, so that product is exact and log2(2^k) == k for every k.
            // Instruction count is unchanged: one multiply plus one multiply-add, where
            // before it was one multiply-add plus one multiply.
            //
            // This is deliberately not applied to other bases.  For log10 the equivalent
            // constant is log10(2) * 2^-23, which is not exactly representable, and
            // folding it in measured worse than the existing path.
            constexpr float TWO_TO_M23 = 1.19209290e-7f;  // 0x1.0p-23
            result = e_float * TWO_TO_M23 + r * sfpi::as<sfpi::vFloat>(sfpi::vUInt(log_base_scale_factor));
        } else {
            result = e_float * sfpi::vConstFloatPrgm0 + r;

            if constexpr (HAS_BASE_SCALING) {
                result *= sfpi::as<sfpi::vFloat>(sfpi::vUInt(log_base_scale_factor));
            }
        }

        // For zero, result is negative before this multiply, so result * +inf
        // gives -inf. For +inf, result is positive, so result * +inf gives
        // +inf. NaNs either skip the main block or propagate here.
        v_if(!sfpi::is_finite(a)) { result *= a; }
        v_endif;
    }
    v_endif;

    return result;
}

bool bf16_dest_log();
template <int ITERATIONS>
void calculate_log_bf16();
void init_log_bf16();
// Whether BF16 DEST runs the generated log kernel as one call over the whole tile.
inline constexpr bool log_bf16_whole_tile = true;
// Sets up the generated BF16 log kernel for the instance it serves.
template <bool bf16_kernel>
inline void log_bf16_tile_init() {
    if constexpr (bf16_kernel) {
        init_log_bf16();
    }
}

template <
    bool APPROXIMATION_MODE,
    bool FAST_APPROX,
    bool HAS_BASE_SCALING,
    bool is_fp32_dest_acc_en,
    int ITERATIONS = 8,
    bool IS_BASE_TWO = false>
inline void calculate_log(uint log_base_scale_factor) {
    if constexpr (!is_fp32_dest_acc_en && !HAS_BASE_SCALING && ITERATIONS == 32) {
        if (bf16_dest_log()) {
            calculate_log_bf16<ITERATIONS>();
            return;
        }
    }
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat result = calculate_log_body<FAST_APPROX, HAS_BASE_SCALING, is_fp32_dest_acc_en, IS_BASE_TWO>(
            sfpi::dst_reg[0], log_base_scale_factor);
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool FAST_APPROX, bool is_fp32_dest_acc_en>
inline void log_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    const float LOG_TWO = 0.693147182f;       // 0x1.62e430p-1
    const float TWO_TO_M23 = 1.19209290e-7f;  // 0x1.0p-23
    // e represents k << 23 rather than k, so pre-fold the 2^(-23) factor into
    // the constant used for the final exponent contribution.
    sfpi::vConstFloatPrgm0 = LOG_TWO * TWO_TO_M23;

    if constexpr (is_fp32_dest_acc_en) {
        // Stored separately because the tuned fp32 m^3 and m^4 coefficients are
        // no longer the shared exact 1/3 and -1/4 values used in the bf16 path.
        sfpi::vConstFloatPrgm1 = -0x1.00001ap-2f;
        sfpi::vConstFloatPrgm2 = 0x1.555572p-2f;
    } else {
        // Horner coefficients used by bf16 polynomial
        sfpi::vConstFloatPrgm1 = 0x1.744p-2f;
        sfpi::vConstFloatPrgm2 = -0x1.008p-1f;
    }
}

}  // namespace sfpu
}  // namespace ckernel

#include "ckernel_sfpu_log_bf16.h"
