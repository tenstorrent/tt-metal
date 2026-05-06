// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_atan2_(sfpi::vFloat y, sfpi::vFloat x) {
    constexpr bool is_bf16 = !is_fp32_dest_acc_en || APPROXIMATION_MODE;

    sfpi::vFloat r;
    sfpi::vFloat q;
    sfpi::vFloat s;

    // Note: if x or y is ±NaN, this ensures that max=NaN, which is important for special case handling.
    sfpi::vFloat min = sfpi::setsgn(x, 0);
    sfpi::vFloat max = sfpi::setsgn(y, 0);
    sfpi::vec_min_max(min, max);

    // a = min(|x|, |y|) / max(|x|, |y|), i.e. a is on [0, 1].
    sfpi::vFloat a = min * sfpu_reciprocal<is_bf16>(max);

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
        sfpi::vFloat c1 = 0x1.99938ap-3f;
        q = __builtin_rvtt_sfpmad(q.get(), s.get(), c1.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        sfpi::vFloat c0 = -0x1.555558p-2f;
        q = __builtin_rvtt_sfpmad(q.get(), s.get(), c0.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    } else {
        q = -0x1.de8p-5f;
        s = a * a;
        sfpi::vFloat c1 = 0x1.668p-3f;
        q = __builtin_rvtt_sfpmad(q.get(), s.get(), c1.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
        sfpi::vFloat c0 = -0x1.54p-2f;
        q = __builtin_rvtt_sfpmad(q.get(), s.get(), c0.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
    }
    sfpi::vFloat half_pi = 0x1.921fb6p+0f;
    sfpi::vFloat t = q * s;
    sfpi::vFloat x_abs = sfpi::setsgn(x, 0);
    r = t * a + a;

    // Special cases:

    v_if(sfpi::reinterpret<sfpi::vInt>(min) >= sfpi::reinterpret<sfpi::vInt>(x_abs)) {
        // if |y| ≥ |x| then r = π/2 - r
        r = half_pi - r;
        v_if(sfpi::reinterpret<sfpi::vInt>(min) >= sfpi::reinterpret<sfpi::vInt>(max)) {
            // if |x| = |y| (including both infinite), then r = π/4
            r = sfpi::addexp(half_pi, -1);
            v_if(min == 0.0f) {
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

    // if sign of x is negative (including x=-0), then r = π - r
    v_if(x < 0.0f) {
        sfpi::vFloat pi = sfpi::addexp(half_pi, 1);
        r = pi - r;
    }
    v_endif;

    if constexpr (!is_fp32_dest_acc_en) {
        r = sfpi::float_to_fp16b(r, sfpi::RoundMode::NearestEven);
    }

    r = sfpi::setsgn(r, y);

    // If |x| = NaN or |y| = NaN, vec_min_max will ensure max=NaN as mentioned above.
    sfpi::vFloat infinity = std::numeric_limits<float>::infinity();
    v_if(sfpi::reinterpret<sfpi::vInt>(infinity) < sfpi::reinterpret<sfpi::vInt>(max)) {
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
    sfpu_reciprocal_init<APPROXIMATION_MODE || !is_fp32_dest_acc_en>();
}

}  // namespace sfpu
}  // namespace ckernel
