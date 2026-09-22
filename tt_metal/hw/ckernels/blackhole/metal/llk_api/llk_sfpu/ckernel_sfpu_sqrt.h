// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_addrmod.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_load_config.h"
#include "sfpu/ckernel_sfpu_rsqrt_compat.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Production constant setup (defined with the init below); declared early so the ITERATIONS != 8
// fallback re-seed inside the calculate function can name it.
template <bool APPROXIMATION_MODE>
inline void _init_sqrt_body_constants_();

// See: Kokosiński, Z., Gepner, P., Moroz, L. et al.
// Fast and accurate approximation algorithms for computing floating point square root. Numerical Algorithms (2024).
// https://doi.org/10.1007/s11075-024-01932-7

// x's bits with the sign shifted out: `_bits_without_sign_(x) != 0` is a magnitude test that
// treats -0.0 and +0.0 alike, which a bare `as<vInt>(x) != 0` does not.
sfpi_inline sfpi::vInt _bits_without_sign_(const sfpi::vFloat x) { return sfpi::as<sfpi::vInt>(x) << 1; }

// Computes the square root or reciprocal square root of a positive floating point value x.
template <bool APPROXIMATE = false, bool RECIPROCAL = false, bool FAST_APPROX = false>
sfpi_inline sfpi::vFloat _calculate_sqrt_body_(const sfpi::vFloat x) {
    sfpi::vInt i = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(x) >> 1);
    sfpi::vFloat y = sfpi::as<sfpi::vFloat>(sfpi::vConstIntPrgm0 - i);

    if constexpr (APPROXIMATE) {
        // Algorithm SQRT_10-bits, with modifications for reciprocal.
        sfpi::vFloat c = x * y;
        sfpi::vFloat negative_y = -y;
        sfpi::vFloat infinity = sfpi::sFloat16b(std::numeric_limits<float>::infinity());
        sfpi::vInt infinity_bits = sfpi::as<sfpi::vInt>(infinity);
        sfpi::vFloat t = sfpi::vConstFloatPrgm1 + negative_y * c;
        if constexpr (RECIPROCAL) {
            sfpi::vInt x_bits = sfpi::as<sfpi::vInt>(x);
            sfpi::vInt infinity_minus_x_bits = infinity_bits - x_bits;
            // If x != inf and x has a non-zero magnitude.
            v_if(infinity_minus_x_bits != 0 && _bits_without_sign_(x) != 0) {
                y = y * t;
                if constexpr (!FAST_APPROX) {
                    // This region already excludes +/-0 and +inf, so a bare sign test is enough.
                    v_if(x < 0.0f) {
                        y = std::numeric_limits<float>::quiet_NaN();  // nan for fp32, inf for bf16
                    }
                    v_endif;
                }
            }
            // Otherwise x = +/-0 gives +/-inf (the subtraction carries x's sign), x = inf gives 0.
            v_else { y = sfpi::as<sfpi::vFloat>(infinity_minus_x_bits); }
            v_endif;
        } else {
            y = c;
            // If x != inf.  Otherwise, y = inf, since c = inf.
            v_if(sfpi::as<sfpi::vInt>(x) != infinity_bits) { y = y * t; }
            v_endif;
        }
    } else {
        // Algorithm SQRT_23-bits, with modifications for reciprocal.
        sfpi::vFloat xy = x * y;
        sfpi::vFloat negative_y = -y;
        sfpi::vFloat c = negative_y * xy;
        sfpi::vFloat infinity = sfpi::sFloat16b(std::numeric_limits<float>::infinity());
        sfpi::vInt infinity_bits = sfpi::as<sfpi::vInt>(infinity);
        y = y * (sfpi::vConstFloatPrgm1 + c * (sfpi::vConstFloatPrgm2 + c));
        xy = x * y;
        negative_y = -y;
        sfpi::vFloat one_minus_xyy = 1.0f + (negative_y * xy);

        if constexpr (RECIPROCAL) {
            sfpi::vFloat half_y = sfpi::addexp(y, -1);
            sfpi::vInt x_bits = sfpi::as<sfpi::vInt>(x);
            sfpi::vInt infinity_minus_x_bits = infinity_bits - x_bits;
            // If x != inf and x has a non-zero magnitude.
            v_if(infinity_minus_x_bits != 0 && _bits_without_sign_(x) != 0) {
                y = one_minus_xyy * half_y + y;
                if constexpr (!FAST_APPROX) {
                    // This region already excludes +/-0 and +inf, so a bare sign test is enough.
                    v_if(x < 0.0f) {
                        y = std::numeric_limits<float>::quiet_NaN();  // nan for fp32, inf for bf16
                    }
                    v_endif;
                }
            }
            // Otherwise x = +/-0 gives +/-inf (the subtraction carries x's sign), x = inf gives 0.
            v_else { y = sfpi::as<sfpi::vFloat>(infinity_minus_x_bits); }
            v_endif;
        } else {
            sfpi::vFloat half_xy = 0.5f * xy;
            // If x == inf, we need to skip to avoid y = inf - inf = nan; y will already be inf.
            // Keep this as `<`, not `!=`: it skips positive NaN, which would otherwise run the
            // step and come back sign-flipped (a bf16 pack then makes that -inf). Negative NaN
            // does run the step -- the compare wraps -- but the clamp below rewrites it.
            v_if(sfpi::as<sfpi::vInt>(x) < infinity_bits) { y = one_minus_xyy * half_xy + xy; }
            v_endif;
        }
    }

    // All edge handling is gated on !FAST_APPROX, as the negative clamp alone was before: the
    // fast path trades every edge guard for speed, so there a negative is unclamped and
    // sqrt(-0) is +0. Everything below is a FAST_APPROX=false claim.
    if constexpr (!FAST_APPROX) {
        // `x < 0.0f` is a sign-bit test, so it claims -0.0 as well; zero magnitudes are kept
        // out of it and answered separately.
        if constexpr (!RECIPROCAL) {
            // rsqrt's clamp is inside the refinement guard above, where the predicate already
            // excludes +/-0 and +inf.
            // sqrt(+/-0) = +/-0: return x, because the refinement cannot produce a signed zero.
            v_if(_bits_without_sign_(x) == 0) { y = x; }
            v_elseif(x < 0.0f) {
                y = std::numeric_limits<float>::quiet_NaN();  // returns nan for fp32 and inf for bf16
            }
            v_endif;
        }
    }

    return y;
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool fp32_dest_acc_en, bool RECIPROCAL, bool FAST_APPROX>
inline void _calculate_sqrt_internal_() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat tmp = _calculate_sqrt_body_<APPROXIMATION_MODE, RECIPROCAL, FAST_APPROX>(sfpi::dst_reg[0]);
        if constexpr (!fp32_dest_acc_en) {
            tmp = sfpi::convert<sfpi::vFloat16b>(tmp, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = tmp;
        sfpi::dst_reg++;
    }
}

// Fast bf16 sqrt for Blackhole. Origin: llk-bench (LLM-agent-written kernel, claude-fable-5, 2026-08),
// validated exhaustively over all 65,536 bf16 inputs vs the exact golden: max 1 ULP (gate <= 2).
// Measured 562.1 cycles/tile vs 882.1 for the previous kernel on p150b (tt-metal v0.76.0 baseline).
// SFPU state programmed by the init: programmable constants vConstIntPrgm0 (LREG12) = 0x5F0B3892 seed magic and
// vConstFloatPrgm1 (LREG13) = -K2 with truncation bias. Pure sfpi (dst_reg[0..7], ADDR_MOD_7 only): no
// SFPLOADMACRO, no replay slots, no ADDR_MOD_6. bf16 DEST only: it relies on the truncating bf16 store instead
// of a rounding op -- gated on !fp32_dest_acc_en below; _calculate_sqrt_body_ stays as the fallback for every
// other configuration. Because sqrt_tile_init() cannot learn FAST_APPROX (only sqrt_tile carries it), this kernel
// also serves FAST_APPROX = true on the bf16 non-approx path; it is exact (max 1 ULP) so that path only gains
// accuracy and edge handling.
//
// Algorithm (SQRT_10-bits family, Kokosinski et al. 2024):
//   y  = as_float(MAGIC - (bits(x) >> 1))     ~ scaled rsqrt seed
//   c  = x * y                                ~ scaled sqrt estimate
//   r  = |c * (y*c - K2)|                     one refinement, ~10-bit accurate
// For finite x, y*c - K2 is negative (~[-1.07,-0.82]) so the final abs
// restores the sign; for x = +inf it is +inf, so the same abs makes
// sqrt(inf) = inf without the predicated fix-up of the sfpi path.
//
// Denormal/zero inputs (exponent field 0): the bit-hack seed is invalid on
// denormals, so those lanes replace x with float(bits(x)) -- exact, since
// bf16-denormal bit patterns are < 2^24 -- and rescale the result by
// sqrt(2)*2^-75 at the end (x = bits * 2^-149 => sqrt(x) = sqrt(bits)*2^-74.5).
// The sign-magnitude SFPCAST maps -0 -> -0.0, and +-0 flow through both
// paths to ordinal-zero outputs. No lane ever feeds a denormal into the FP
// pipeline (the SFPU MAD path flushes denormal operands).
//
// DST is bf16, so the final SFPSTORE truncates fp32 -> bf16. Instead of a
// round-nearest convert op, K2 carries a +1.5e-3 multiplicative bias
// (~+0.4 bf16 ULP) centering the truncation error. Exhaustive simulation of
// this exact pipeline over all 65536 bf16 inputs gives max 1 ULP vs the
// float64 golden (gate: 2 ULP); verified on silicon.
inline void _init_sqrt_bf16_fast_() {
    // Common SFPU init (config reg + ADDR_MOD_7 + counter reset) inlined so this init is self-contained; the
    // kernel only uses ADDR_MOD_7 (sfpi dst_reg accesses).
    _init_sfpu_config_reg();
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    math::reset_counters(p_setrwc::SET_ABD_F);

    sfpi::vConstIntPrgm0 = 0x5F0B3892;     // rsqrt seed magic
    sfpi::vConstFloatPrgm1 = -1.8938275f;  // -K2 * (1 + 1.5e-3 trunc bias)
}

// One face (8 dst vectors).
inline void _calculate_sqrt_bf16_fast_() {
    constexpr int vectors_per_face = 8;
#pragma GCC unroll 8
    for (int d = 0; d < vectors_per_face; d++) {
        sfpi::vFloat x = sfpi::dst_reg[d];
        sfpi::vInt e = sfpi::exexp(x, sfpi::ExponentMode::Biased);
        v_if(e == 0) { x = sfpi::convert<sfpi::vFloat>(sfpi::as<sfpi::vSMag>(x), sfpi::RoundMode::Nearest); }
        v_endif;
        sfpi::vInt i = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(x) >> 1);
        sfpi::vFloat y = sfpi::as<sfpi::vFloat>(sfpi::vConstIntPrgm0 - i);
        sfpi::vFloat c = x * y;
        sfpi::vFloat t = y * c + sfpi::vConstFloatPrgm1;
        sfpi::vFloat r = c * t;
        v_if(e == 0) { r = r * 0x1.6ap-75f; }  // == bf16(sqrt(2)*2^-75), emits SFPMULI
        v_endif;
        sfpi::dst_reg[d] = sfpi::abs(r);
    }
}

template <
    bool APPROXIMATION_MODE,
    int ITERATIONS = 8,
    bool fp32_dest_acc_en,
    bool FAST_APPROX,
    bool legacy_compat = false>
inline void calculate_sqrt() {
    // FAST_APPROX is deliberately not part of the gate: see _init_sqrt_bf16_fast_ (the init cannot know it).
    if constexpr (!APPROXIMATION_MODE && !legacy_compat && !fp32_dest_acc_en && ITERATIONS == 8) {
        _calculate_sqrt_bf16_fast_();
        return;
    }
    if constexpr (
        !legacy_compat && !APPROXIMATION_MODE && !fp32_dest_acc_en &&
        !(!APPROXIMATION_MODE && !legacy_compat && !fp32_dest_acc_en && ITERATIONS == 8)) {
        // ITERATIONS != 8 (tt-llk tests): sqrt_init<false, false> cannot see ITERATIONS and has programmed the
        // fast kernel's vConstIntPrgm0 / vConstFloatPrgm1 (and left vConstFloatPrgm2 unset) on top of the
        // SQRT_23-bits constants _calculate_sqrt_body_ reads; re-seed them before falling back.
        _init_sqrt_body_constants_<APPROXIMATION_MODE>();
    }
    if constexpr (legacy_compat) {
        _calculate_sqrt_compat_<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en>(ITERATIONS);
    } else {
        _calculate_sqrt_internal_<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en, false, FAST_APPROX>();
    }
}

// SQRT_10/23-bits seed + refinement constants read by _calculate_sqrt_body_ (the sfpi sqrt/rsqrt paths and body
// re-users: asin/acos endpoint sqrt, add_rsqrt). Body re-users call this directly: sqrt_init / rsqrt_init below
// program the fast bf16 kernels' constants instead on their gates.
template <bool APPROXIMATION_MODE>
inline void _init_sqrt_body_constants_() {
    if constexpr (APPROXIMATION_MODE) {
        sfpi::vConstIntPrgm0 = 0x5f0b3892;
        sfpi::vConstFloatPrgm1 = 1.89099014875f;
    } else {
        sfpi::vConstIntPrgm0 = 0x5f1110a0;
        sfpi::vConstFloatPrgm1 = 2.2825186f;
        sfpi::vConstFloatPrgm2 = 2.2533049f;
    }
}

// is_fp32_dest_acc_en selects between the fast bf16 kernel's constants and the production ones; ITERATIONS is
// not threaded here (same convention as recip_init), every caller instantiates calculate_sqrt with 8.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool legacy_compat = false>
void sqrt_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (!legacy_compat) {
        if constexpr (!APPROXIMATION_MODE && !is_fp32_dest_acc_en) {
            _init_sqrt_bf16_fast_();
        } else {
            _init_sqrt_body_constants_<APPROXIMATION_MODE>();
        }
    }
}

}  // namespace sfpu
}  // namespace ckernel
