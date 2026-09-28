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
// max_iter = 2: sufficient for float32 precision (≤1 ulps).
// max_iter = 1: sufficient for bfloat16/float16 precision (≤0.5 ulps).
// max_iter = 0: this has the same effect as max_iter=1 at the moment;
//               it may be replaced with a cheaper approximation in future.
template <int max_iter = 2, bool round_to_bf16 = false>
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
    // SFPNOT also flips the sign; multiplying by -0.5 below restores it.
    // Not only is 255-in.Exp more efficient via SFPNOT, but it also ensures
    // that in.Exp == 0 results in ±inf, and in.Exp == 255 results in ±0.
    // See the scale factor adjustment via scale*(-0.5) below for further details.
    sfpi::vUInt scale_bits = ~sfpi::as<sfpi::vUInt>(in);

    // Continue with quadratic estimate.
    y = sfpi::vConstFloatPrgm2 + y * negative_x;

    // Scale factor: set mantissa to zero.
    sfpi::vFloat scale = sfpi::setman(sfpi::as<sfpi::vFloat>(scale_bits), 0);

    // First iteration of Newton-Raphson: t = 1.0 - x*y.
    sfpi::vFloat t = 1.0f + negative_x * y;

    // Scale factor adjustment: halve the magnitude and restore the input sign.
    // If scale = ±inf, then scale*(-0.5) = ∓inf and scale.Exp=255.
    // If scale = ±0, then scale*(-0.5) = 0 and scale.Exp=0.
    // Otherwise, scale.Exp = scale.Exp-1 = 255-in.Exp-1 = 254-in.Exp.
    scale *= -0.5f;

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
            out = sfpu_reciprocal_iter<1, true>(in);
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
    // Fit y = k2 - k1*x + k0*x**2 over [1,2), constraining the one-step
    // Newton result to the correct BF16 rounding intervals. With Wormhole
    // FMA semantics, all 128 normalized BF16 inputs round correctly after
    // one step; two steps remain faithful for all FP32 mantissas (<0.890 ULP).
    sfpi::vConstFloatPrgm0 = 0.32133400440216064453125f;
    sfpi::vConstFloatPrgm1 = 1.4514148235321044921875f;
    sfpi::vConstFloatPrgm2 = 2.1200883388519287109375f;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_reciprocal() {
    _calculate_reciprocal_internal_<APPROXIMATION_MODE, ITERATIONS, is_fp32_dest_acc_en>(ITERATIONS);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en /*maybe_unused*/>
void recip_init() {
    // Common SFPU init inlined (SFPU config register + ADDR_MOD_7 + counter reset), then the op-specific
    // reciprocal setup below -- one self-contained init, matching exp_init. SDPA runs reciprocal in its
    // softmax after matmul/exp, so the general SFPU state is re-established here, not just reset.
    // Reciprocal uses only ADDR_MOD_7 on Wormhole (no op-specific ADDR_MOD_6).
    sfpu::_init_sfpu_config_reg();
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel
