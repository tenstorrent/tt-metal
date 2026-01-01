// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_atan2_(sfpi::vFloat y, sfpi::vFloat x) {
    constexpr bool is_bf16 = !is_fp32_dest_acc_en || APPROXIMATION_MODE;

    sfpi::vFloat r;
    sfpi::vFloat q;
    sfpi::vFloat s;
    sfpi::vFloat y_abs;
    sfpi::vFloat x_abs;

    // Note: if x or y is ±NaN, the use of setsgn, ensures that
    // max=NaN, which abs does not.  This is important for special
    // case handling.
    auto [min, max] = sfpi::min_max(sfpi::setsgn(x, 0), sfpi::setsgn(y, 0));

    sfpi::vFloat scale = sfpi::setman(sfpi::as<sfpi::vFloat>(~sfpi::as<sfpi::vUInt>(max)), 0);
    sfpi::vFloat m = sfpi::setexp(max, 128);
    sfpi::vFloat scaled_min = min * scale;
    sfpi::vFloat recip = sfpi::approx_recip(m);
    if constexpr (!is_bf16) {
        sfpi::vFloat e = m * recip - 2.0f;
        recip = recip * -e - 0.0f;
        e = m * recip - 2.0f;
        recip = recip * -e - 0.0f;
    }
    sfpi::vFloat a = scaled_min * recip;

    // Next we compute the minimax approximation for atan(a).

    if constexpr (is_fp32_dest_acc_en) {
        q = 0x1.01cp-8f;
        s = a * a;
        sfpi::vFloat c6 = -0x1.4bcp-6f;
        q = __builtin_rvtt_sfpmad(q.get(), s.get(), c6.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        sfpi::vFloat c5 = 0x1.93p-5f;
        q = __builtin_rvtt_sfpmad(q.get(), s.get(), c5.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        sfpi::vFloat c4 = -0x1.48cp-4f;
        q = __builtin_rvtt_sfpmad(q.get(), s.get(), c4.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        sfpi::vFloat c3 = 0x1.bd4p-4f;
        q = __builtin_rvtt_sfpmad(q.get(), s.get(), c3.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        sfpi::vFloat c2 = -0x1.24p-3f;
        q = __builtin_rvtt_sfpmad(q.get(), s.get(), c2.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        if constexpr (!is_bf16) {
            y_abs = sfpi::setsgn(y, 0);
        }
        sfpi::vFloat c1 = sfpi::vConstFloatPrgm1;
        q = __builtin_rvtt_sfpmad(q.get(), s.get(), c1.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        x_abs = sfpi::setsgn(x, 0);
        sfpi::vFloat c0 = sfpi::vConstFloatPrgm2;
        q = __builtin_rvtt_sfpmad(q.get(), s.get(), c0.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    } else {
        q = -0x1.de8p-5f;
        s = a * a;
        sfpi::vFloat c1 = 0x1.668p-3f;
        q = __builtin_rvtt_sfpmad(q.get(), s.get(), c1.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        sfpi::vFloat c0 = -0x1.54p-2f;
        q = __builtin_rvtt_sfpmad(q.get(), s.get(), c0.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        x_abs = sfpi::setsgn(x, 0);
    }
    sfpi::vFloat neg_half_pi = sfpi::vConstFloatPrgm0;
    sfpi::vFloat t = q * s;
    r = t * a + a;

    // Special cases:
    sfpi::vFloat y_side;
    sfpi::vFloat x_side;
    if constexpr (is_bf16) {
        y_side = min;
        x_side = min;
    } else {
        y_side = y_abs;
        x_side = x_abs;
    }

    v_if(sfpi::as<sfpi::vInt>(y_side) >= sfpi::as<sfpi::vInt>(x_abs)) {
        r = neg_half_pi - r;
        v_if(sfpi::as<sfpi::vInt>(x_side) >= sfpi::as<sfpi::vInt>(max)) {
            r = sfpi::addexp(neg_half_pi, -1);
            v_if(max == 0.0f) {
                // if both zero, then r = ±0
                // SFPI note: the later v_if(x < 0.0f) behaves like a signbit check, so r=-0
                // is handled by that path.
                r = 0.0f;
            }
            v_endif;
        }
        v_endif;
    }
    v_endif;

    v_if(x < 0.0f) {
        sfpi::vFloat neg_pi;
        if constexpr (!is_fp32_dest_acc_en) {
            neg_pi = sfpi::vConstFloatPrgm2;
        } else {
            neg_pi = sfpi::addexp(neg_half_pi, 1);
        }
        r = neg_pi - r;
    }
    v_endif;

    if constexpr (!is_fp32_dest_acc_en) {
        r = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
    }

    r = sfpi::copysgn(r, y);

    // If |x| = NaN or |y| = NaN, vec_min_max will ensure max=NaN as mentioned above.
    sfpi::vFloat infinity;
    if constexpr (!is_fp32_dest_acc_en) {
        infinity = sfpi::vConstFloatPrgm1;
    } else {
        infinity = std::numeric_limits<float>::infinity();
    }
    v_if(sfpi::as<sfpi::vInt>(infinity) < sfpi::as<sfpi::vInt>(max)) {
        // if |x| = NaN or |y| = NaN, then r = NaN
        r = std::numeric_limits<float>::quiet_NaN();
    }
    v_endif;

    return r;
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_atan2(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr uint dst_tile_size_sfpi = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat result = _sfpu_atan2_<APPROXIMATION_MODE, is_fp32_dest_acc_en>(in0, in1);

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_atan2_init() {
    sfpi::vConstFloatPrgm0 = -0x1.921fb6p+0f;
    if constexpr (is_fp32_dest_acc_en) {
        sfpi::vConstFloatPrgm1 = 0x1.99938ap-3f;
        sfpi::vConstFloatPrgm2 = -0x1.555558p-2f;
    } else {
        sfpi::vConstFloatPrgm1 = std::numeric_limits<float>::infinity();
        sfpi::vConstFloatPrgm2 = -0x1.921fb6p+1f;
    }
}

}  // namespace sfpu
}  // namespace ckernel
