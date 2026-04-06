// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_softsign() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];

        // Clamp input to avoid subnormal intermediates in reciprocal.
        // For very large |x|, 1/(1+|x|) becomes subnormal and hardware
        // flushes it to zero, giving softsign(x)=0 instead of ~sign(x).
        // Clamping to 1e30 is safe: softsign(1e30) rounds to +-1.0 in
        // both bfloat16 and float32.
        v_if(v > 1e30f) { v = 1e30f; }
        v_endif;
        v_if(v < -1e30f) { v = -1e30f; }
        v_endif;

        sfpi::vFloat tmp = sfpi::abs(v) + sfpi::vConst1;
        tmp = sfpu_reciprocal<APPROXIMATION_MODE>(tmp);
        sfpi::dst_reg[0] = v * tmp;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void init_softsign() {
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
