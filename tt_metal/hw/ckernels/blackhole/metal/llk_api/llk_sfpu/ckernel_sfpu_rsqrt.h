// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_sqrt.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_load_config.h"
#include "sfpu/ckernel_sfpu_rsqrt_compat.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Fast bf16 rsqrt for Blackhole. Origin: llk-bench (LLM-agent-written kernel, claude-opus-5, 2026-08),
// validated exhaustively over all 65,536 bf16 inputs vs the exact golden: max 1 ULP (gate <= 2).
// Measured 626.1 cycles/tile on p150b; the bench run for this op recorded no cycles/tile for the previous
// kernel -- measure _calculate_sqrt_body_<false, true, *> when validating on hardware.
// SFPU state programmed by the init: programmable constants vConstIntPrgm0 (LREG12) = seed magic,
// vConstFloatPrgm1 (LREG13) = C1, vConstFloatPrgm2 (LREG14) = 2^74.5 subnormal rescale. Pure sfpi (dst_reg[0..7],
// ADDR_MOD_7 only): no SFPLOADMACRO, no replay slots, no ADDR_MOD_6. bf16 DEST only: the output is left truncated
// by the bf16 store -- gated on !fp32_dest_acc_en below; _calculate_sqrt_body_ stays as the fallback for every
// other configuration. Because rsqrt_tile_init<legacy_compat>() cannot learn FAST_APPROX (only rsqrt_tile carries
// it), this kernel also serves FAST_APPROX = true on the bf16 non-approx path (exact, max 1 ULP; negative inputs
// are don't-care on both).
//
// Core is the "SQRT_10-bits" bit trick plus its single correction step, which
// is ~10 bits accurate -- ample for a 2-ULP bf16 result, so the output is left
// truncated by SFPSTORE instead of paying for a round-to-nearest:
//
//   y0 = as_float(K - (bits(u) >> 1))          ~ 0.77 * rsqrt(u)
//   y  = (y0 * s) * (C1 - y0 * (u * y0))
//
// u = |x| normally.  SFPMAD flushes subnormal operands to zero, so subnormal
// lanes instead use u = float(bits(|x|)) = |x| * 2^149 (one SFPCAST -- exact,
// and always normal), with s = 2^74.5 undoing the scaling at the end; s = 1
// elsewhere.  Note SFPCAST must not be issued in place on Blackhole, which is
// why u is written from a distinct source register.
//
// Edge cases fall out of that same path:
//   +-0   -> exponent 0, so u = +0, y0 = as_float(K) ~ 1e19 and y0 * s
//            overflows to +inf; the closing setsgn turns -0 into -inf.
//   +inf  -> u * y0 = inf, so C1 - y0*c = -inf; max(.,0) turns it into +0.
//   x < 0 -> don't-care (golden is NaN); |x| keeps them finite.
//
// Two vectors are processed per iteration so the compiler interleaves
// the two dependency chains and hides SFPU result latency.
static constexpr int RSQRT_FAST_K = 0x5f0b3892;
static constexpr float RSQRT_FAST_C1 = 1.89099014875f;
static constexpr float RSQRT_FAST_SCALE = 2.671373890628154e22f;  // 2^74.5

inline void _init_rsqrt_bf16_fast_() {
    // Common SFPU init (config reg + ADDR_MOD_7 + counter reset) inlined so this init is self-contained; the
    // kernel only uses ADDR_MOD_7 (sfpi dst_reg accesses).
    _init_sfpu_config_reg();
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    math::reset_counters(p_setrwc::SET_ABD_F);

    sfpi::vConstIntPrgm0 = RSQRT_FAST_K;
    sfpi::vConstFloatPrgm1 = RSQRT_FAST_C1;
    sfpi::vConstFloatPrgm2 = RSQRT_FAST_SCALE;
}

sfpi_inline void _rsqrt_fast_step_(sfpi::vFloat& u, sfpi::vFloat& s) {
    v_if(sfpi::exexp(u, sfpi::ExponentMode::Biased) == 0) {  // subnormal or zero
        u = sfpi::vFloat(
            __builtin_rvtt_sfpcast(sfpi::as<sfpi::vInt>(u).get(), sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE));
        s = sfpi::vConstFloatPrgm2;
    }
    v_endif;
}

sfpi_inline sfpi::vFloat _rsqrt_fast_guess_(sfpi::vFloat u) {
    return sfpi::as<sfpi::vFloat>(
        sfpi::vConstIntPrgm0 - sfpi::as<sfpi::vInt>(sfpi::shft(sfpi::as<sfpi::vUInt>(u), -1)));
}

// One face (8 dst vectors), two at a time.
inline void _calculate_rsqrt_bf16_fast_() {
    constexpr size_t vectors_per_face = 8;
#pragma GCC unroll 4
    for (size_t i = 0; i < vectors_per_face; i += 2) {
        sfpi::vFloat x0 = sfpi::dst_reg[i];
        sfpi::vFloat x1 = sfpi::dst_reg[i + 1];
        sfpi::vFloat u0 = sfpi::abs(x0);
        sfpi::vFloat u1 = sfpi::abs(x1);
        sfpi::vFloat s0 = sfpi::vFloat(1.0f);
        sfpi::vFloat s1 = sfpi::vFloat(1.0f);

        _rsqrt_fast_step_(u0, s0);
        _rsqrt_fast_step_(u1, s1);

        sfpi::vFloat y0 = _rsqrt_fast_guess_(u0);
        sfpi::vFloat y1 = _rsqrt_fast_guess_(u1);
        sfpi::vFloat c0 = u0 * y0;
        sfpi::vFloat c1 = u1 * y1;
        sfpi::vFloat p0 = sfpi::vConstFloatPrgm1 - y0 * c0;
        sfpi::vFloat p1 = sfpi::vConstFloatPrgm1 - y1 * c1;
        sfpi::vFloat z0 = y0 * s0;
        sfpi::vFloat z1 = y1 * s1;

        // sfpi >= 7.80: setsgn(vFloat, vInt) removed; copysgn lowers to the same SFPSETSGN (sign bit of x).
        sfpi::dst_reg[i] = sfpi::copysgn(z0 * sfpi::max(p0, 0.0f), sfpi::as<sfpi::vInt>(x0));
        sfpi::dst_reg[i + 1] = sfpi::copysgn(z1 * sfpi::max(p1, 0.0f), sfpi::as<sfpi::vInt>(x1));
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool fp32_dest_acc_en, bool FAST_APPROX, bool legacy_compat>
inline void calculate_rsqrt() {
    // FAST_APPROX is deliberately not part of the gate: see _init_rsqrt_bf16_fast_ (the init cannot know it).
    if constexpr (!APPROXIMATION_MODE && !legacy_compat && !fp32_dest_acc_en && ITERATIONS == 8) {
        _calculate_rsqrt_bf16_fast_();
        return;
    }
    if constexpr (
        !legacy_compat && !APPROXIMATION_MODE && !fp32_dest_acc_en &&
        !(!APPROXIMATION_MODE && !legacy_compat && !fp32_dest_acc_en && ITERATIONS == 8)) {
        // ITERATIONS != 8 (tt-llk tests): rsqrt_init<false, false, false> cannot see ITERATIONS and has programmed
        // the fast kernel's vConstIntPrgm0 / vConstFloatPrgm1 / vConstFloatPrgm2 on top of the SQRT_23-bits
        // constants _calculate_sqrt_body_ reads (via _calculate_sqrt_internal_); re-seed them before delegating.
        _init_sqrt_body_constants_<APPROXIMATION_MODE>();
    }
    if constexpr (legacy_compat) {
        _calculate_rsqrt_compat_<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en>(ITERATIONS);
    } else {
        _calculate_sqrt_internal_<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en, true, FAST_APPROX>();
    }
}

// is_fp32_dest_acc_en selects between the fast bf16 kernel's constants and the production ones (shared with
// sqrt, see _init_sqrt_body_constants_); ITERATIONS is not threaded here (same convention as recip_init).
template <bool APPROXIMATION_MODE, bool legacy_compat, bool is_fp32_dest_acc_en>
void rsqrt_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (!legacy_compat) {
        if constexpr (!APPROXIMATION_MODE && !is_fp32_dest_acc_en) {
            _init_rsqrt_bf16_fast_();
        } else {
            _init_sqrt_body_constants_<APPROXIMATION_MODE>();
        }
    }
}

}  // namespace sfpu
}  // namespace ckernel
