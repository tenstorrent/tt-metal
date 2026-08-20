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
// for what it costs and why it exists.
template <bool APPROXIMATION_MODE, bool GUARD_NON_FINITE = true>
sfpi_inline sfpi::vFloat sfpu_sqrt_custom(sfpi::vFloat in) {
    sfpi::vFloat val = in;
    sfpi::vFloat out = val;
    // Zero and the non-finite inputs all pass `val` straight through, which is already the
    // answer for +/-0 and +inf. Non-finite has to be excluded because the seed above is
    // ~5.2e-20 for +inf; squaring it underflows to a denormal, SFPMAD flushes that to +0,
    // and the next multiply is 0 * -inf = NaN -- so sqrt_custom(+inf) was NaN, and every
    // consumer inherited it (erfinv(+/-1)). See issue #52930.
    //
    // Tested on the biased exponent field rather than a compare against infinity because
    // SFPSETCC's float compare is specified only for inputs that are not NaN
    // (VectorUnit.md), and this predicate has to be evaluated on a possible NaN. SFPI's `&&`
    // is SFPXBOOL(AND) and does not short-circuit, so the `val != 0.0f` compare is still
    // emitted for NaN lanes -- that is safe, not merely tolerated: AND is monotone and
    // exexp(NaN) == 255 makes the second conjunct false, so the conjunction is false for
    // those lanes whatever the unspecified compare returned, and NaN passes through.
    //
    // -inf is a known residual: it passes through as -inf where IEEE and the golden give
    // NaN. Synthesising a NaN for negative input is deliberately not done here -- erfinv's
    // NR undershoot drives `tmp + intermediate_result` (ckernel_sfpu_erfinv.h:40)
    // non-positive for small in-domain x, so a negative-to-NaN guard would turn ordinary
    // inputs such as erfinv(1e-6) into NaN. Tracked under issue #52930; needs its own fix.
    if constexpr (GUARD_NON_FINITE) {
        v_if(val != 0.0f && sfpi::exexp(val, sfpi::ExponentMode::Biased) != 255) { out = sfpu_sqrt_custom_newton(val); }
        v_endif;
    } else {
        // The guard costs 3 SFPU instructions per vector iteration (SFPEXEXP, SFPIADD,
        // SFPSETCC) = +96 cycles/tile. Only opt out where the argument provably cannot be
        // non-finite on any lane whose result is committed.
        v_if(val != 0.0f) { out = sfpu_sqrt_custom_newton(val); }
        v_endif;
    }
    return out;
}

}  // namespace sfpu
}  // namespace ckernel
