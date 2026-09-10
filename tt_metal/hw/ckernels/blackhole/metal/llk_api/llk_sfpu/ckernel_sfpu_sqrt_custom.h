// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// NEGATIVE_INFINITY_SAFE trades a branch for IEEE sqrt(-inf) = NaN; see the note on the guard
// at the bottom of this function for why it is off by default.
template <bool APPROXIMATION_MODE, bool NEGATIVE_INFINITY_SAFE = false>
sfpi_inline sfpi::vFloat sfpu_sqrt_custom(sfpi::vFloat in) {
    sfpi::vFloat val = in;
    sfpi::vFloat out = val;
    // Skipped lanes pass `val` through, already the answer for +/-0 and +inf. Non-finite needs
    // excluding because the +inf seed (~5.2e-20) squares to a denormal, SFPMAD flushes it to +0,
    // and 0 * -inf = NaN: sqrt_custom(+inf) was NaN and consumers inherited it (erfinv(+/-1)).
    //
    // Exponent test rather than a compare against inf: SFPSETCC's float compare is unspecified
    // for NaN (VectorUnit.md), and `&&` is SFPXBOOL(AND), so `val != 0.0f` is still evaluated on
    // NaN lanes. That is safe -- is_finite(NaN) is false and AND is monotone, so NaN passes
    // through whatever the compare returned.
    v_if(val != 0.0f && sfpi::is_finite(val)) {
        // Fast inverse square-root seed + two Newton-Raphson refinements.
        sfpi::vUInt magic = sfpi::as<sfpi::vUInt>(sfpi::vFloat(sfpi::sFloat16b(0x5f37)));
        sfpi::vFloat approx = sfpi::as<sfpi::vFloat>(magic - (sfpi::as<sfpi::vUInt>(val) >> 1));
        sfpi::vFloat neg_half_val = val * -0.5f;
        approx = ((approx * approx) * neg_half_val + 1.5f) * approx;
        approx = ((approx * approx) * neg_half_val + 1.5f) * approx;
        out = approx * val;
    }
    v_endif;

    // IEEE sqrt(-inf) = NaN, where the pass-through above yields -inf. Opt-in: no production
    // consumer can reach a -inf and the branch is not free (erfinv paid ~1.12x for it on WH,
    // this arch unmeasured), so only the test-only calculate_sqrt_custom wrapper turns it on.
    // Must stay an exact-pattern test against a literal +qNaN -- reasoning with the edge sweep.
    if constexpr (NEGATIVE_INFINITY_SAFE) {
        v_if(sfpi::as<sfpi::vInt>(val) == static_cast<int>(0xFF800000)) {
            out = std::numeric_limits<float>::quiet_NaN();
        }
        v_endif;
    }
    return out;
}

}  // namespace sfpu
}  // namespace ckernel
