// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// Fast inverse square-root seed + two Newton-Raphson refinements. Callers reach this through
// sfpu_sqrt_custom, which is where the domain guard lives; on its own this is valid only for
// finite non-zero input.
sfpi_inline sfpi::vFloat sfpu_sqrt_custom_newton(sfpi::vFloat val) {
    sfpi::vUInt magic = sfpi::as<sfpi::vUInt>(sfpi::vFloat(sfpi::sFloat16b(0x5f37)));
    sfpi::vFloat approx = sfpi::as<sfpi::vFloat>(magic - (sfpi::as<sfpi::vUInt>(val) >> 1));
    sfpi::vFloat neg_half_val = val * -0.5f;
    approx = ((approx * approx) * neg_half_val + 1.5f) * approx;
    approx = ((approx * approx) * neg_half_val + 1.5f) * approx;
    return approx * val;
}

// GUARD_NON_FINITE excludes non-finite input from the iteration. Leave it on unless the call
// site can prove no lane ever reaches here with a 255 exponent -- see the guard comment below
// for why it exists.
template <bool APPROXIMATION_MODE, bool GUARD_NON_FINITE = true>
sfpi_inline sfpi::vFloat sfpu_sqrt_custom(sfpi::vFloat in) {
    sfpi::vFloat val = in;
    sfpi::vFloat out = val;
    // Skipped lanes pass `val` through, already the answer for +/-0 and +inf. Non-finite needs
    // excluding because the +inf seed (~5.2e-20) squares to a denormal, SFPMAD flushes it to +0,
    // and 0 * -inf = NaN: sqrt_custom(+inf) was NaN and consumers inherited it (erfinv(+/-1)).
    //
    // Exponent test rather than a compare against inf: SFPSETCC's float compare is unspecified
    // for NaN (VectorUnit.md), and `&&` is SFPXBOOL(AND), so `val != 0.0f` is still evaluated on
    // NaN lanes. That is safe -- exexp(NaN) == 255 falsifies the other conjunct and AND is
    // monotone, so NaN passes through whatever the compare returned.
    //
    // Residual: -inf passes through where IEEE and the golden give NaN. No negative-to-NaN guard,
    // because erfinv's NR undershoot makes `tmp + intermediate_result` (ckernel_sfpu_erfinv.h:40)
    // non-positive for small in-domain x, which would turn erfinv(1e-6) into NaN.
    if constexpr (GUARD_NON_FINITE) {
        v_if(val != 0.0f && sfpi::exexp(val, sfpi::ExponentMode::Biased) != 255) { out = sfpu_sqrt_custom_newton(val); }
        v_endif;
    } else {
        // Only opt out where the argument provably cannot be non-finite on any lane whose result
        // is committed.
        v_if(val != 0.0f) { out = sfpu_sqrt_custom_newton(val); }
        v_endif;
    }
    return out;
}

}  // namespace sfpu
}  // namespace ckernel
