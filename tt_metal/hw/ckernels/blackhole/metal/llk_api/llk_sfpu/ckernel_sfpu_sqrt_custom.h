// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// NEGATIVE_INFINITY_SAFE adds a branch for IEEE sqrt(-inf) = NaN; off by default, see below.
// NEWTON_ITERATIONS only exists so the parameters match Wormhole's; 2 is the value this arch
// has always used, and with a constant bound the loop unrolls to the two steps it replaces.
template <bool APPROXIMATION_MODE, int NEWTON_ITERATIONS = 2, bool NEGATIVE_INFINITY_SAFE = false>
sfpi_inline sfpi::vFloat sfpu_sqrt_custom(sfpi::vFloat in) {
    sfpi::vFloat val = in;
    sfpi::vFloat out = val;
    // Skipped lanes pass `val` through, already the answer for +/-0 and +inf. Non-finite needs
    // excluding because the +inf seed (~5.2e-20) squares to a denormal, SFPMAD flushes it to +0,
    // and 0 * -inf = NaN: sqrt_custom(+inf) was NaN and consumers inherited it (erfinv(+/-1)).
    //
    // NaN lanes still evaluate `val != 0.0f`, whose result is unspecified (VectorUnit.md), but
    // is_finite(NaN) is false and `&&` is a monotone AND, so they pass through regardless.
    v_if(val != 0.0f && sfpi::is_finite(val)) {
        // Fast inverse square-root seed + Newton-Raphson refinements.
        sfpi::vUInt magic = sfpi::as<sfpi::vUInt>(sfpi::vFloat(sfpi::sFloat16b(0x5f37)));
        sfpi::vFloat approx = sfpi::as<sfpi::vFloat>(magic - (sfpi::as<sfpi::vUInt>(val) >> 1));
        sfpi::vFloat neg_half_val = val * -0.5f;
#pragma GCC unroll 2
        for (int i = 0; i < NEWTON_ITERATIONS; i++) {
            approx = ((approx * approx) * neg_half_val + 1.5f) * approx;
        }
        out = approx * val;
    }
    v_endif;

    // IEEE sqrt(-inf) = NaN, where the pass-through above yields -inf. Opt-in because it is not
    // free (always-on costs erfinv ~1.13x) and no production consumer can reach a -inf; only the
    // test-only calculate_sqrt_custom wrapper turns it on.
    //
    // Exact bit pattern, and a literal +qNaN: `val < 0.0f` would claim -0.0 too (SFPSETCC reads
    // the sign bit) and a sign-carrying NaN would be negative, which a bf16 pack makes -inf.
    if constexpr (NEGATIVE_INFINITY_SAFE) {
        constexpr std::int32_t negative_infinity_bits = static_cast<std::int32_t>(0xFF800000);
        v_if(sfpi::as<sfpi::vInt>(val) == negative_infinity_bits) { out = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
    }
    return out;
}

}  // namespace sfpu
}  // namespace ckernel
