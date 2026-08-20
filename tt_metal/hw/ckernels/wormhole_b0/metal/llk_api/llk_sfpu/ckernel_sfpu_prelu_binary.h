// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// prelu(a, w) = a < 0 ? a * w : a, with w per element rather than a scalar.
// The body is the same as calculate_prelu's; only where the weight comes from
// differs, so the two stay one line apart on purpose.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_prelu_binary_(sfpi::vFloat a, sfpi::vFloat w) {
    v_if(a < 0.0f) { a = a * w; }
    v_endif;

    if constexpr (!is_fp32_dest_acc_en) {
        a = sfpi::convert<sfpi::vFloat16b>(a, sfpi::RoundMode::Nearest);
    }

    return a;
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_prelu_binary(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr uint dst_tile_size_sfpi = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat result = _sfpu_prelu_binary_<APPROXIMATION_MODE, is_fp32_dest_acc_en>(in0, in1);

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_prelu_binary_init() {}

}  // namespace sfpu
}  // namespace ckernel
