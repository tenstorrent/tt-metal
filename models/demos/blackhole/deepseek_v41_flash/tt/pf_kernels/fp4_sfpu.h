// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// SFPU parts of the fused fp4 (e2m1, per-32 e8m0 scale) quantise-dequantise of tt/pf_tune.fp4_fast.

#if defined(TRISC_MATH)

#include "sfpi.h"

namespace ckernel::sfpu {

// dst tile holds amax (row max of |x| over the 32 columns, in column 0).  MODE 0: 1 / scale, MODE 1: scale, with
// scale = 2^ceil(log2(max(amax, 6 * 2^-126) * fp32(1/6)))  (exact powers of two).
template <int MODE>
inline void fp4_scale_sfpu() {
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat a = sfpi::dst_reg[0];
        v_if(a < 0x1.8p-124f) { a = 0x1.8p-124f; }
        v_endif;
        sfpi::vFloat v = a * 0.16666667163372040f;
        sfpi::vInt e = sfpi::exexp(v);  // unbiased exponent
        sfpi::vFloat p = sfpi::setexp(v, 127);
        v_if(p > 1.0f) { e += 1; }
        v_endif;
        if constexpr (MODE == 0) {
            sfpi::dst_reg[0] = sfpi::setexp(1.0f, 127 - e);
        } else {
            sfpi::dst_reg[0] = sfpi::setexp(1.0f, e + 127);
        }
        sfpi::dst_reg++;
    }
}

// dst tile holds y = x / scale: round |y| to the e2m1 grid {0, .5, 1, 1.5, 2, 3, 4, 6} (half up, clamp at 6), keep the
// sign.
inline void fp4_grid_sfpu() {
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat y = sfpi::dst_reg[0];
        sfpi::vFloat m = sfpi::abs(y);
        sfpi::vFloat o = 6.0f;
        v_if(m < 5.0f) { o = 4.0f; }
        v_endif;
        v_if(m < 3.5f) { o = 3.0f; }
        v_endif;
        v_if(m < 2.5f) { o = 2.0f; }
        v_endif;
        v_if(m < 1.75f) { o = 1.5f; }
        v_endif;
        v_if(m < 1.25f) { o = 1.0f; }
        v_endif;
        v_if(m < 0.75f) { o = 0.5f; }
        v_endif;
        v_if(m < 0.25f) { o = 0.0f; }
        v_endif;
        v_if(y < 0.0f) { o = -o; }
        v_endif;
        sfpi::dst_reg[0] = o;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu

#endif
