// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_sqrt.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// Numerically-stable hypot: m * sqrt(1 + (n/m)^2), m=max(|a|,|b|), n=min(|a|,|b|).
// The naive sqrt(a^2 + b^2) form overflows for |x| > 2^64 and underflows for
// 0 < |x| < 2^-63 even when hypot itself is finite normal. The rescale keeps the
// intermediate `1 + (n/m)^2` in [1, 2] regardless of the magnitude of m.
//
// Both `1/m` and `sqrt(1 + r^2)` are computed via the shared rsqrt polynomial
// (sqrt_init constants): `1/m = rsqrt(m)^2` avoids needing a separate reciprocal
// init that would clobber the sqrt constants.
//
// Register-pressure notes (SFPU has no spill slot — any spill is a fatal compile
// error): raw_a and raw_b are consumed by min_max immediately, and the NaN/inf
// guards later derive their information from m and n (not the raw inputs), so no
// extra live values are carried across the two sqrt-body calls. Scoped blocks
// tell the compiler when intermediates die.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_hypot_(sfpi::vFloat a, sfpi::vFloat b) {
    // setsgn preserves NaN payload while zeroing the sign; guards below detect
    // NaN via bits(m) > bits(+inf), which requires the setsgn form (abs alone
    // does not preserve NaN).
    auto [n, m] = sfpi::min_max(sfpi::setsgn(a, 0), sfpi::setsgn(b, 0));

    // Compute the rescaled ratio n/m, its square, and 1 + r^2 in a scoped block
    // so rsqrt_m and r die before the second sqrt-body call. Both sqrt-body
    // calls MUST use the same APPROXIMATION_MODE — the polynomial constants
    // configured by sqrt_init are keyed on APPROXIMATION_MODE, so mixing modes
    // silently uses the wrong constants and produces garbage.
    sfpi::vFloat s;
    {
        sfpi::vFloat rsqrt_m = _calculate_sqrt_body_<APPROXIMATION_MODE, /*RECIPROCAL=*/true, /*FAST_APPROX=*/true>(m);
        // 1/m = rsqrt(m)^2 — this is why we can skip a separate reciprocal init.
        sfpi::vFloat r = n * (rsqrt_m * rsqrt_m);
        s = 1.0f + r * r;
    }

    sfpi::vFloat sqrt_s = _calculate_sqrt_body_<APPROXIMATION_MODE, /*RECIPROCAL=*/false, /*FAST_APPROX=*/true>(s);

    sfpi::vFloat result = m * sqrt_s;

    // Special cases. Order: 0/0 → 0 first (baseline); then NaN; then inf (inf
    // wins over NaN per IEEE 754 hypot).
    v_if(m == 0.0f) { result = 0.0f; }
    v_endif;

    sfpi::vFloat infinity = std::numeric_limits<float>::infinity();
    sfpi::vInt inf_bits = sfpi::as<sfpi::vInt>(infinity);

    // NaN in max: any input was NaN.
    v_if(sfpi::as<sfpi::vInt>(m) > inf_bits) {
        // IEEE 754 hypot(inf, NaN) = +inf. min_max with (inf, NaN) yields
        // (inf, NaN), so n == +inf signals the disambiguated inf case.
        v_if(sfpi::as<sfpi::vInt>(n) == inf_bits) { result = infinity; }
        v_else { result = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
    }
    v_endif;

    // m is exactly +inf: at least one input was ±inf, no NaN present.
    v_if(sfpi::as<sfpi::vInt>(m) == inf_bits) { result = infinity; }
    v_endif;

    if constexpr (!is_fp32_dest_acc_en) {
        result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
    }

    return result;
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_hypot(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr uint dst_tile_size_sfpi = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat result = _sfpu_hypot_<APPROXIMATION_MODE, is_fp32_dest_acc_en>(in0, in1);

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_hypot_init() {
    // Reuses the sqrt polynomial constants (vConstIntPrgm0, vConstFloatPrgm1/2).
    // Both `1/m` (via rsqrt^2) and `sqrt(1+r^2)` share this single init.
    sqrt_init<APPROXIMATION_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel
