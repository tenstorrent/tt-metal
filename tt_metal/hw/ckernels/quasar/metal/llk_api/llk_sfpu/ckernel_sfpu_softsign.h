// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_recip.h"
#include "cmath_common.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_softsign() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        // Clamp |v| at 2**26: from there on v / (1 + |v|) rounds to +-1.0f in float32, and the clamp keeps
        // 1 / (1 + |v|) far from the smallest normal, below which it flushes to zero. poison = v - v is NaN for +-inf
        // and NaN inputs and 0 otherwise, so those inputs still produce NaN; the sign goes on before the multiply,
        // so the result is never re-signed and a NaN keeps the sign the hardware gives it.
        sfpi::vFloat va = sfpi::min(sfpi::setsgn(v, 0), sfpi::vFloat(0x1.0p26f));
        sfpi::vFloat denom = va + 1.0f;
        sfpi::vFloat poison = v - v;
        sfpi::dst_reg[0] = sfpi::copysgn(va, v) * _sfpu_reciprocal_<(APPROXIMATION_MODE) ? 0 : 2>(denom) + poison;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void init_softsign() {
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
    _init_reciprocal_<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
