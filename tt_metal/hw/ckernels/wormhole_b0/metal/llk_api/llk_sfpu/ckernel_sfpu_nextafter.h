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

// nextafter(a, b) is the representable value adjacent to a in the direction of
// b, so it is defined on the bit pattern rather than on the value: step the
// magnitude by one and put the sign back. That is one pass over dst, which is
// why this does not decompose into arithmetic ops without either losing the
// step at large magnitudes or overshooting at small ones.
//
// The step is one unit in the LAST place of the destination format, not of the
// register. Registers are fp32 whatever dst holds, so a bfloat16 destination
// steps by 0x10000, which is also exactly its smallest subnormal.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_nextafter_(sfpi::vFloat a, sfpi::vFloat b) {
    constexpr int step = is_fp32_dest_acc_en ? 1 : 0x10000;
    constexpr float infinity = std::numeric_limits<float>::infinity();

    // The magnitude grows when the direction of travel agrees with the sign of
    // a. Comparing signs through copysgn rather than the product a * (b - a)
    // keeps it correct when that product would overflow or flush.
    sfpi::vFloat d = b - a;
    sfpi::vFloat toward = sfpi::copysgn(sfpi::vFloat(1.0f), a) * sfpi::copysgn(sfpi::vFloat(1.0f), d);

    sfpi::vInt m = sfpi::as<sfpi::vInt>(sfpi::setsgn(a, 0));
    v_if(toward > 0.0f) { m = m + step; }
    v_else { m = m - step; }
    v_endif;
    sfpi::vFloat r = sfpi::copysgn(sfpi::as<sfpi::vFloat>(m), a);

    // a infinite needs no case of its own: the magnitude of an infinity is the
    // exponent field all ones with a zero mantissa, so one step down is the
    // largest finite value, which is the answer, and one step up cannot be
    // asked for because that would need b beyond infinity.

    // a zero has no magnitude bits to walk down, and its neighbour in either
    // direction is the smallest subnormal carrying the sign of b.
    v_if(sfpi::setsgn(a, 0) == 0.0f) {
        r = sfpi::copysgn(sfpi::as<sfpi::vFloat>(sfpi::vInt(step)), b);
    }
    v_endif;

    // IEEE-754: nextafter(x, x) is x, for every x including the zeroes.
    v_if(a == b) { r = b; }
    v_endif;

    sfpi::vFloat vinf = infinity;
    v_if(sfpi::as<sfpi::vInt>(sfpi::setsgn(a, 0)) > sfpi::as<sfpi::vInt>(vinf)) { r = a; }
    v_endif;
    v_if(sfpi::as<sfpi::vInt>(sfpi::setsgn(b, 0)) > sfpi::as<sfpi::vInt>(vinf)) { r = b; }
    v_endif;

    if constexpr (!is_fp32_dest_acc_en) {
        r = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
    }

    return r;
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_nextafter(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr uint dst_tile_size_sfpi = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat result = _sfpu_nextafter_<APPROXIMATION_MODE, is_fp32_dest_acc_en>(in0, in1);

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_nextafter_init() {}

}  // namespace sfpu
}  // namespace ckernel
