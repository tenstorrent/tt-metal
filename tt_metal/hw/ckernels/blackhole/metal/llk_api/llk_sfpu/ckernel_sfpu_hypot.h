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

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_hypot_(sfpi::vFloat a, sfpi::vFloat b) {
    auto [n, m] = sfpi::min_max(sfpi::setsgn(a, 0), sfpi::setsgn(b, 0));

    sfpi::vFloat s;
    {
        sfpi::vFloat rsqrt_m = _calculate_sqrt_body_<APPROXIMATION_MODE, /*RECIPROCAL=*/true, /*FAST_APPROX=*/true>(m);
        sfpi::vFloat r = n * (rsqrt_m * rsqrt_m);
        s = 1.0f + r * r;
    }

    sfpi::vFloat sqrt_s = _calculate_sqrt_body_<APPROXIMATION_MODE, /*RECIPROCAL=*/false, /*FAST_APPROX=*/true>(s);

    sfpi::vFloat result = m * sqrt_s;

    v_if(m == 0.0f) { result = 0.0f; }
    v_endif;

    sfpi::vFloat infinity = std::numeric_limits<float>::infinity();
    sfpi::vInt inf_bits = sfpi::as<sfpi::vInt>(infinity);

    v_if(sfpi::as<sfpi::vInt>(m) > inf_bits) {
        v_if(sfpi::as<sfpi::vInt>(n) == inf_bits) { result = infinity; }
        v_else { result = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
    }
    v_endif;

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
    sqrt_init<APPROXIMATION_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel
