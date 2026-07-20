// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_rsqrt_compat.h"
using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Computes the reciprocal of a floating point value x.
// max_iter specifies the number of Newton-Raphson iterations.
// max_iter = 2: sufficient for float32 precision (≤1 ulps).
// max_iter = 1: sufficient for bfloat16/float16 precision (≤0.5 ulps).
// max_iter = 0: this has the same effect as max_iter=1 at the moment;
//               it may be replaced with a cheaper approximation in future.
template <int max_iter = 2>
sfpi_inline sfpi::vFloat sfpu_reciprocal_iter(const sfpi::vFloat in) {
    // Combines the sign and exponent of -1.0 with the mantissa of `in`.
    // Scale the input value to the range [1.0, 2.0), and make it negative.
    // If in ≠ ±0 and in ≠ ±inf, then x = in * 2**(127-in.Exp).
    // If in = ±0 or in = ±inf, then x = ±1.
    // Then negative_x = -x.
    sfpi::vFloat negative_x = sfpi::copyman(-1.0f, in);

    // Quadratic initial estimate: y = k2 - k1*x + k0*x**2.
    sfpi::vFloat y = sfpi::vConstFloatPrgm1 + sfpi::vConstFloatPrgm0 * negative_x;

    // Scale factor: we want 1/in = 1/x * scale.
    // For x ≠ ±0 and x ≠ ±inf, in = x * 2**-(127-in.Exp), so 1/in = 1/x * 2**(127-in.Exp).
    // Add float32 bias: scale.Exp = 127+127-in.Exp = 254-in.Exp.
    // For efficiency and handling of x = ±0 and x = ±inf, we set scale.Exp = 255-in.Exp = ~in.Exp.
    // This is efficiently computed with a single SFPNOT, followed by SFPSETMAN to clear the mantissa at the next
    // opportunity.
    // The sign doesn't matter as we set the output sign to match the input at the end.
    // Not only is 255-in.Exp more efficient via SFPNOT, but it also ensures
    // that in.Exp == 0 results in ±inf, and in.Exp == 255 results in ±0.
    // See the scale factor adjustment via scale*0.5 below for further details.
    sfpi::vUInt scale_bits = ~sfpi::as<sfpi::vUInt>(in);

    // Continue with quadratic estimate.
    y = sfpi::vConstFloatPrgm2 + y * negative_x;

    // Scale factor: set mantissa to zero.
    sfpi::vFloat scale = sfpi::setman(sfpi::as<sfpi::vFloat>(scale_bits), 0);

    // First iteration of Newton-Raphson: t = 1.0 - x*y.
    sfpi::vFloat t = 1.0f + negative_x * y;

    // Scale factor adjustment: scale = scale*0.5.
    // If scale = ±inf, then scale*0.5 = ±inf and scale.Exp=255.
    // If scale = ±0, then scale*0.5 = 0 and scale.Exp=0.
    // Otherwise, scale.Exp = scale.Exp-1 = 255-in.Exp-1 = 254-in.Exp.
    scale *= 0.5f;

    // Continue Newton-Raphson: y = y + y*t.
    y = y + y * t;

    if constexpr (max_iter > 1) {
        // Second iteration of Newton-Raphson: t = 1.0 - x*y; y = y + y*t.
        t = 1.0f + negative_x * y;
        y = y + y * t;
    }

    // Apply scaling factor, and set sign to match input.
    y = y * scale;
    y = sfpi::copysgn(y, in);

    return y;
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void _calculate_reciprocal_internal_(const int iterations) {
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat out;

        if constexpr (APPROXIMATION_MODE) {
            out = sfpu_reciprocal_iter<0>(in);
        } else if constexpr (is_fp32_dest_acc_en) {
            out = sfpu_reciprocal_iter<2>(in);
        } else {
            out = sfpu_reciprocal_iter<1>(in);
            out = sfpi::convert<sfpi::vFloat16b>(out, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = out;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATE = false, bool save_reg = true /* Unused. Enough registers available. */>
sfpi_inline vFloat sfpu_reciprocal(const vFloat in) {
    return sfpu_reciprocal_iter<APPROXIMATE ? 0 : 2>(in);
}

template <bool APPROXIMATE = false>
sfpi_inline void sfpu_reciprocal_init() {
    // The polynomial y = k2 - k1*x + k0*x**2 minimises the maximum
    // relative error for 1/x over the interval [1,2), via Sollya.
    sfpi::vConstFloatPrgm0 = 0.3232325017452239990234375f;
    sfpi::vConstFloatPrgm1 = 1.4545459747314453125f;
    sfpi::vConstFloatPrgm2 = 2.121212482452392578125f;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8, bool legacy_compat = false>
inline void calculate_reciprocal() {
    if constexpr (legacy_compat) {
        _calculate_reciprocal_compat_<APPROXIMATION_MODE, ITERATIONS, is_fp32_dest_acc_en>(ITERATIONS);
    } else {
        _calculate_reciprocal_internal_<APPROXIMATION_MODE, ITERATIONS, is_fp32_dest_acc_en>(ITERATIONS);
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool legacy_compat = false>
void recip_init() {
    // Common SFPU init inlined (ADDR_MOD_7 + counter reset), then the op-specific
    // reciprocal setup below -- one self-contained init, matching exp_init. SDPA runs reciprocal in its
    // softmax after matmul/exp, so the general SFPU state is re-established here, not just reset.
    // Reciprocal uses only ADDR_MOD_7 on Wormhole (no op-specific ADDR_MOD_6).
    // NOTE: the SFPU config register is programmed once per kernel by the hoisted llk_math_sfpu_init_once(),
    // not per op. On Blackhole, re-programming it per op corrupts the fp32 SFPLOADMACRO reciprocal (#50381);
    // Wormhole reciprocal has no such path, but the once-per-kernel invariant is kept consistent across arches.
    // Only the op's ADDR_MOD state + counter reset are re-established here.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (!legacy_compat) {
        sfpu_reciprocal_init<APPROXIMATION_MODE>();
    }
}

}  // namespace sfpu
}  // namespace ckernel
