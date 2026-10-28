// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpi.h"
using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Computes the reciprocal of a floating point value x.
// max_iter specifies the number of Newton-Raphson iterations.
// max_iter = 2: sufficient for faithful FP32 rounding in the normal input/output domain.
// max_iter = 1: with round_to_bf16 and the matching opt-in seed from sfpu_reciprocal_init,
//               correctly rounds BF16 inputs with normal BF16 reciprocals (≤0.5 ULP).
// max_iter = 0: this has the same effect as max_iter=1 at the moment;
//               it may be replaced with a cheaper approximation in future.
// round_to_bf16 rounds the normalized result to BF16, nearest with ties away from zero,
// before power-of-two scaling. Intended for BF16 Dest; this preserves the normal
// reciprocal ±2^-126 at input ±2^126 instead of flushing an intermediate result.
// normalized=true requires a finite input with magnitude in [1, 2) and skips scaling.
template <int max_iter = 2, bool round_to_bf16 = false, bool normalized = false>
sfpi_inline sfpi::vFloat sfpu_reciprocal_iter(const sfpi::vFloat in) {
    // Combines the sign and exponent of -1.0 with the mantissa of `in`.
    // Scale the input value to the range [1.0, 2.0), and make it negative.
    // If in ≠ ±0 and in ≠ ±inf, then x = in * 2**(127-in.Exp).
    // If in = ±0 or in = ±inf, then x = ±1.
    // Then negative_x = -x.
    sfpi::vFloat negative_x;
    if constexpr (normalized) {
        // The exponent is already 127; only the sign needs changing.
        negative_x = sfpi::setsgn(in, 1);
    } else {
        negative_x = sfpi::copyman(-1.0f, in);
    }

    // Quadratic initial estimate: y = k2 - k1*x + k0*x**2.
    sfpi::vFloat y = sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0 * negative_x;

    // Scale factor: we want 1/in = 1/x * scale.
    // For x ≠ ±0 and x ≠ ±inf, in = x * 2**-(127-in.Exp), so 1/in = 1/x * 2**(127-in.Exp).
    // Add float32 bias: scale.Exp = 127+127-in.Exp = 254-in.Exp.
    // For efficiency and handling of x = ±0 and x = ±inf, we set scale.Exp = 255-in.Exp = ~in.Exp.
    // This is efficiently computed with a single SFPNOT, followed by SFPSETMAN to clear the mantissa at the next
    // opportunity.
    // Not only is 255-in.Exp more efficient via SFPNOT, but it also ensures
    // that in.Exp == 0 results in ±inf, and in.Exp == 255 results in ±0.
    // See the scale factor adjustment via scale*0.5 below for further details.
    sfpi::vUInt scale_bits;
    if constexpr (!normalized) {
        scale_bits = ~sfpi::as<sfpi::vUInt>(in);
    }

    // Continue with quadratic estimate.
    y = sfpi::vConstFloatPrgm2 + y * negative_x;

    // Scale factor: set mantissa to zero.
    sfpi::vFloat scale;
    if constexpr (!normalized) {
        scale = sfpi::setman(sfpi::as<sfpi::vFloat>(scale_bits), 0);
    }

    // First iteration of Newton-Raphson: t = 1.0 - x*y.
    sfpi::vFloat t = 1.0f + negative_x * y;

    // Scale factor adjustment: halve the magnitude.
    // If scale = ±inf, then scale*0.5 = ±inf and scale.Exp=255.
    // If scale = ±0, then scale*0.5 = 0 and scale.Exp=0.
    // Otherwise, scale.Exp = scale.Exp-1 = 255-in.Exp-1 = 254-in.Exp.
    if constexpr (!normalized) {
        scale *= 0.5f;
    }

    // Continue Newton-Raphson: y = y + y*t.
    y = y + y * t;

    if constexpr (max_iter > 1) {
        // Second iteration of Newton-Raphson: t = 1.0 - x*y; y = y + y*t.
        t = 1.0f + negative_x * y;
        y = y + y * t;
    }

    if constexpr (round_to_bf16) {
        // Round before scaling: for in == +/-2**126, an unrounded value
        // just below 1 would otherwise underflow to zero instead of giving
        // +/-2**-126. Power-of-two scaling preserves BF16 precision.
        y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
    }

    // Apply scaling factor and restore the sign, including for zero results.
    // Wormhole multiplication discards the sign of zero. Preserve it here
    // even when a subsequent BF16 pack currently discards it again.
    if constexpr (!normalized) {
        y = y * scale;
    }
    y = sfpi::copysgn(y, in);

    return y;
}

constexpr std::uint32_t RECIP_SEQ_NEG_MANT = 0x0084;
constexpr std::uint32_t RECIP_SEQ_SCALE_BF16 = 0x2705;
constexpr std::uint32_t RECIP_SEQ_SCALE_FP32_A = 0x3705;
constexpr std::uint32_t RECIP_SEQ_SCALE_FP32_B = 0x3f05;
constexpr std::uint32_t RECIP_SEQ_SIGN_STORE_LO = 0x00c6;
constexpr std::uint32_t RECIP_SEQ_SIGN_STORE_HI = 0x4b00;
// UnitDelayKind = elapsed instructions; set on entry and cleared on exit of each call (see #46751).
constexpr std::uint32_t RECIP_MACRO_MISC = 0xf40;

inline void reciprocal_bf16_macro_config() {
    TTI_SFPSETMAN(0, p_sfpu::LCONST_neg1, 12, 0);
    TTI_SFPNOT(0, 0, 13, 0);
    TTI_SFPSETSGN(0, p_sfpu::LREG3, 14, 0);
    TTI_SFPMULI(0x3f00, 15, 0);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, RECIP_SEQ_NEG_MANT);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, 0x0000);
    TTI_SFPCONFIG(0, 4 + 0, 0);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, RECIP_SEQ_SCALE_BF16);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, 0x0000);
    TTI_SFPCONFIG(0, 4 + 1, 0);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, RECIP_SEQ_SIGN_STORE_LO);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, RECIP_SEQ_SIGN_STORE_HI);
    TTI_SFPCONFIG(0, 4 + 2, 0);
}

inline void reciprocal_bf16_macro_head() {
    TTI_SFPLOADMACRO((0 << 2) | 0, 0, ADDR_MOD_3, 0);
    TTI_SFPLOADMACRO((0 << 2) | 0, 0, ADDR_MOD_3, 2 | 1);
    TTI_SFPMAD(p_sfpu::LREG12, p_sfpu::LREG0, p_sfpu::LREG13, p_sfpu::LREG2, 0);
    TTI_SFPMAD(p_sfpu::LREG12, p_sfpu::LREG4, p_sfpu::LREG13, p_sfpu::LREG6, 0);
    TTI_SFPLOADMACRO((1 << 2) | 1, 0, ADDR_MOD_3, 0);
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LREG14, p_sfpu::LREG2, 0);
    TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG4, p_sfpu::LREG14, p_sfpu::LREG6, 0);
    TTI_SFPSETMAN(0, p_sfpu::LREG1, p_sfpu::LREG1, 1);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG0, 0);
    TTI_SFPLOADMACRO((1 << 2) | 1, 0, ADDR_MOD_3, 2 | 1);
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG6, p_sfpu::LCONST_1, p_sfpu::LREG4, 0);
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG2, 0);
    TTI_SFPSETMAN(0, p_sfpu::LREG5, p_sfpu::LREG5, 1);
    TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG4, p_sfpu::LREG6, p_sfpu::LREG6, 0);
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN,
        0,
        p_sfpu::LREG0,
        p_sfpu::LREG2,
        p_sfpu::LREG2,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN,
        0,
        p_sfpu::LREG0,
        p_sfpu::LREG6,
        p_sfpu::LREG6,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
}

inline void reciprocal_bf16_macro_tail() {
    TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
    TTI_SFPLOADMACRO((2 << 2) | 1, 0, ADDR_MOD_3, 0);
    TTI_SFPMUL(p_sfpu::LREG6, p_sfpu::LREG5, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
    TTI_SFPLOADMACRO((2 << 2) | 1, 0, ADDR_MOD_2, 2 | 1);
}

inline void reciprocal_fp32_macro2_config() {
    TTI_SFPSETMAN(0, p_sfpu::LCONST_neg1, 12, 0);
    TTI_SFPNOT(0, 0, 13, 0);
    TTI_SFPSETSGN(0, p_sfpu::LREG3, 14, 0);
    TTI_SFPMULI(0x3f00, 15, 0);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, RECIP_SEQ_NEG_MANT);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, 0x0000);
    TTI_SFPCONFIG(0, 4 + 0, 0);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, RECIP_SEQ_SCALE_FP32_A);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, 0x0000);
    TTI_SFPCONFIG(0, 4 + 1, 0);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, RECIP_SEQ_SIGN_STORE_LO);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, RECIP_SEQ_SIGN_STORE_HI);
    TTI_SFPCONFIG(0, 4 + 2, 0);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, RECIP_SEQ_SCALE_FP32_B);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, 0x0000);
    TTI_SFPCONFIG(0, 4 + 3, 0);
}

inline void reciprocal_fp32_macro2_head() {
    TTI_SFPLOADMACRO((0 << 2) | 0, 0, ADDR_MOD_3, 2 | 1);
    TTI_SFPMAD(p_sfpu::LREG12, p_sfpu::LREG0, p_sfpu::LREG13, p_sfpu::LREG2, 0);
    TTI_SFPMAD(p_sfpu::LREG12, p_sfpu::LREG4, p_sfpu::LREG13, p_sfpu::LREG6, 0);
    TTI_SFPLOADMACRO((1 << 2) | 1, 0, ADDR_MOD_3, 0);
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LREG14, p_sfpu::LREG2, 0);
    TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG4, p_sfpu::LREG14, p_sfpu::LREG6, 0);
    TTI_SFPSETMAN(0, p_sfpu::LREG1, p_sfpu::LREG1, 1);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG3, 0);
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG6, p_sfpu::LCONST_1, p_sfpu::LREG7, 0);
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG2, 0);
    TTI_SFPLOADMACRO((3 << 2) | 1, 0, ADDR_MOD_3, 2 | 1);
    TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG7, p_sfpu::LREG6, p_sfpu::LREG6, 0);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG3, 0);
    TTI_SFPSETMAN(0, p_sfpu::LREG5, p_sfpu::LREG5, 1);
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG6, p_sfpu::LCONST_1, p_sfpu::LREG7, 0);
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG2, 0);
}

inline void reciprocal_fp32_macro2_tail_a() {
    TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG7, p_sfpu::LREG6, p_sfpu::LREG6, 0);
    TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
    TTI_SFPLOADMACRO((2 << 2) | 1, 0, ADDR_MOD_3, 0);
}

inline void reciprocal_fp32_macro2_tail_b() {
    TTI_SFPMUL(p_sfpu::LREG6, p_sfpu::LREG5, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
    TTI_SFPLOADMACRO((2 << 2) | 1, 0, ADDR_MOD_2, 2 | 1);
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void _calculate_reciprocal_internal_(const int iterations) {
#ifdef DISABLE_SFPLOADMACRO
    constexpr bool use_macro = false;
#else
    constexpr bool use_macro = true;
#endif
    if constexpr (APPROXIMATION_MODE || (ITERATIONS & 1) || !use_macro) {
#pragma GCC unroll 8
        for (int d = 0; d < iterations; d++) {
            sfpi::vFloat in = sfpi::dst_reg[0];
            sfpi::vFloat out;
            if constexpr (APPROXIMATION_MODE) {
                out = sfpu_reciprocal_iter<0>(in);
            } else if constexpr (is_fp32_dest_acc_en) {
                out = sfpu_reciprocal_iter<2>(in);
            } else {
                out = sfpu_reciprocal_iter<1 /*max_iter*/, true /*round_to_bf16*/>(in);
            }
            sfpi::dst_reg[0] = out;
            sfpi::dst_reg++;
        }
    } else if constexpr (is_fp32_dest_acc_en) {
        TTI_SFPCONFIG(RECIP_MACRO_MISC, 8, 1);
        TTI_SFPNOP;
        TTI_SFPLOADMACRO((0 << 2) | 0, 0, ADDR_MOD_3, 0);
        TTI_REPLAY(0, 16, 1, 1);
        reciprocal_fp32_macro2_head();
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS / 2 - 1; d++) {
            if (d > 0) {
                TTI_REPLAY(0, 16, 0, 0);
            }
            reciprocal_fp32_macro2_tail_a();
            TTI_SFPLOADMACRO((0 << 2) | 0, 0, ADDR_MOD_3, 4);
            reciprocal_fp32_macro2_tail_b();
        }
        if constexpr (ITERATIONS / 2 > 1) {
            TTI_REPLAY(0, 16, 0, 0);
        }
        reciprocal_fp32_macro2_tail_a();
        TTI_SFPNOP;
        reciprocal_fp32_macro2_tail_b();
        TTI_SFPNOP;
        TTI_SFPNOP;
        TTI_SFPCONFIG(0, 8, 1);
        TTI_SFPNOP;
        TTI_SFPNOP;
    } else {
        TTI_SFPCONFIG(RECIP_MACRO_MISC, 8, 1);
        TTI_SFPNOP;
        TTI_REPLAY(0, 16, 1, 1);
        reciprocal_bf16_macro_head();
        reciprocal_bf16_macro_tail();
#pragma GCC unroll 8
        for (int d = 1; d < ITERATIONS / 2; d++) {
            TTI_REPLAY(0, 16, 0, 0);
            reciprocal_bf16_macro_tail();
        }
        TTI_SFPNOP;
        TTI_SFPNOP;
        TTI_SFPCONFIG(0, 8, 1);
        TTI_SFPNOP;
        TTI_SFPNOP;
    }
}

template <bool APPROXIMATE = false, bool save_reg = true /* Unused. Enough registers available. */>
sfpi_inline vFloat sfpu_reciprocal(const vFloat in) {
    return sfpu_reciprocal_iter<APPROXIMATE ? 0 : 2>(in);
}

template <bool APPROXIMATE = false, bool round_to_bf16 = false>
sfpi_inline void sfpu_reciprocal_init() {
    if constexpr (round_to_bf16) {
        // Fit y = k2 - k1*x + k0*x**2 over [1,2), constraining the one-step
        // Newton result to the correct BF16 rounding intervals. All 128
        // normalized BF16 inputs round correctly with Wormhole FMA semantics.
        // Not shared with the FP32 path: the minimax seed below has maximum
        // relative error 0.0101, but BF16 correct rounding needs at most
        // 0.0039 near x = 2, which forces a maximum of at least 0.0118 on
        // [1,2). Two Newton steps raise that error to the fourth power, so
        // a shared seed would increase FP32 error, e.g. in polygamma.
        sfpi::vConstFloatPrgm0 = 0.32133400440216064453125f;
        sfpi::vConstFloatPrgm1 = 1.4514148235321044921875f;
        sfpi::vConstFloatPrgm2 = 2.1200883388519287109375f;
    } else {
        // Shared minimax seed for unrounded one- and two-step reciprocals,
        // including intermediate reciprocals in compound BF16 operations.
        sfpi::vConstFloatPrgm0 = 0.3232325017452239990234375f;
        sfpi::vConstFloatPrgm1 = 1.4545459747314453125f;
        sfpi::vConstFloatPrgm2 = 2.121212482452392578125f;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_reciprocal() {
    _calculate_reciprocal_internal_<APPROXIMATION_MODE, ITERATIONS, is_fp32_dest_acc_en>(ITERATIONS);
}

// BF16 rounding is an explicit opt-in: compound callers may pass false for
// is_fp32_dest_acc_en even when their reciprocal intermediate is FP32.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en /*maybe_unused*/, bool round_to_bf16 = false>
void recip_init() {
    // Common SFPU init inlined (SFPU config register + ADDR_MOD_7 + counter reset), then the op-specific
    // reciprocal setup below -- one self-contained init, matching exp_init. SDPA runs reciprocal in its
    // softmax after matmul/exp, so the general SFPU state is re-established here, not just reset.
    // Also programs ADDR_MOD_6 and the load macros; call it again after another op's init.
    sfpu::_init_sfpu_config_reg();
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<APPROXIMATION_MODE, round_to_bf16>();
#ifndef DISABLE_SFPLOADMACRO
    if constexpr (!APPROXIMATION_MODE && (round_to_bf16 || is_fp32_dest_acc_en)) {
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_6);
    }
    if constexpr (!APPROXIMATION_MODE && round_to_bf16) {
        reciprocal_bf16_macro_config();
    }
    if constexpr (!APPROXIMATION_MODE && is_fp32_dest_acc_en) {
        reciprocal_fp32_macro2_config();
    }
#endif
}

}  // namespace sfpu
}  // namespace ckernel
