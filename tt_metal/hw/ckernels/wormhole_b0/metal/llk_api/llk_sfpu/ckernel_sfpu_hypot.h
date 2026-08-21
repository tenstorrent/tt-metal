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

// hypot(a, b) = sqrt(a^2 + b^2), evaluated after scaling the pair by a power of
// two so that neither square can leave the format. The scale is exact, so the
// answer is the one the plain formula would give in a wider format.
//
// The plain formula needs |x| < 2^64 to keep x^2 finite and |x| > 2^-63 to keep
// x^2 normal. Everything outside that band is brought into it by one multiply,
// which keeps the single square root the plain formula already had.
//
// inf and NaN are settled before the arithmetic rather than repaired after it:
// hypot(inf, NaN) is +inf, so the inf is promoted into the maximum and every
// other special case then falls out of the square root itself. Nothing but the
// undo scale stays live across the square root, which is what keeps the SFPU
// from spilling.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_hypot_(sfpi::vFloat a, sfpi::vFloat b) {
    // setsgn preserves the NaN payload while zeroing the sign, so a NaN stays
    // distinguishable from +inf by its bits below.
    auto [n, m] = sfpi::min_max(sfpi::setsgn(a, 0), sfpi::setsgn(b, 0));

    sfpi::vFloat infinity = std::numeric_limits<float>::infinity();

    // IEEE 754 hypot(inf, NaN) = +inf. n == +inf can only happen when the other
    // operand is +inf or NaN, so promoting m here settles both.
    v_if(sfpi::as<sfpi::vInt>(n) == sfpi::as<sfpi::vInt>(infinity)) { m = infinity; }
    v_endif;

    sfpi::vFloat undo = 1.0f;
    v_if(m > 0x1p63f) {
        m = m * 0x1p-100f;
        n = n * 0x1p-100f;
        undo = 0x1p100f;
    }
    v_elseif(m < 0x1p-63f) {
        m = m * 0x1p100f;
        n = n * 0x1p100f;
        undo = 0x1p-100f;
    }
    v_endif;

    sfpi::vFloat result =
        _calculate_sqrt_body_<APPROXIMATION_MODE, /*RECIPROCAL=*/false, /*FAST_APPROX=*/true>(m * m + n * n) * undo;

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
    sqrt_init<APPROXIMATION_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel
