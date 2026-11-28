// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_ldexp_(sfpi::vFloat a, sfpi::vFloat b) {
    sfpi::vFloat f, j, r, s, t, abs_b;
    sfpi::vInt i;
    sfpi::vSMag16 sm;

    sm = sfpi::convert<sfpi::vSMag16>(b, sfpi::RoundMode::Nearest);
    j = sfpi::convert<sfpi::vFloat>(sm, sfpi::RoundMode::Nearest);
    f = b - j;

    r = 0x1.41cp-13f;
    r = r * f + 0x1.5f4p-10f;
    r = r * f + 0x1.3b4p-7f;
    i = sfpi::abs(sfpi::as<sfpi::vInt>(sm));
    s = r * f + sfpi::vConstFloatPrgm2;
    i = sfpi::as<sfpi::vInt>(sfpi::copysgn(sfpi::as<sfpi::vFloat>(i), j));
    r = s * f + sfpi::vConstFloatPrgm1;
    s *= std::numeric_limits<float>::infinity();
    r = r * f + sfpi::vConstFloatPrgm0;
    abs_b = sfpi::abs(b);
    r = r * f + 1.0f;
    t = 1.0f;

    v_if(abs_b >= 0.0f) {
        sfpi::vInt e = sfpi::exexp(r, sfpi::ExponentMode::Biased);
        e += i;
        v_if(sfpi::nearby(e < 255)) {
            s = sfpi::setexp(r, e);
            v_if(sfpi::nearby(e < 1)) {
                s = sfpi::setexp(r, e + 64);
                t = 0x1p-64f;
                v_if(e < -63) { s = 0.0f; }
                v_endif;
            }
            v_endif;
        }
        v_endif;
    }
    v_endif;
    v_if(b <= -150.0f) { t = 0.0f; }
    v_endif;

    sfpi::vFloat y = a * s * t;
    if constexpr (!is_fp32_dest_acc_en) {
        y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
    }
    return y;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_sfpu_ldexp(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr uint dst_tile_size_sfpi = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat b = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = _sfpu_ldexp_<is_fp32_dest_acc_en>(a, b);
        sfpi::dst_reg++;
    }
}

inline void calculate_sfpu_ldexp_init() {
    sfpi::vConstFloatPrgm0 = 0x1.62e42ep-1f;
    sfpi::vConstFloatPrgm1 = 0x1.ebfba0p-3f;
    sfpi::vConstFloatPrgm2 = 0x1.c6afd8p-5f;
}

}  // namespace ckernel::sfpu
