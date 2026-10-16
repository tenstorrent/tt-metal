// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_conversions.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// prelu(a, w) = a < 0 ? a * w : a, with w per element rather than a scalar. It gives the
// composite it replaces, where(ltz(a), multiply(a, w), a), bit for bit: a lane takes the
// product only where ltz would be true (sign set, not +/-0.0, not NaN), and the product is
// rounded and zeroed as multiply rounds and zeroes it.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_prelu_binary_(sfpi::vFloat a, sfpi::vFloat w) {
    const sfpi::vInt abs_bits = sfpi::as<sfpi::vInt>(sfpi::setsgn(a, 0));
    v_if(a < 0.0f && abs_bits != 0 && abs_bits <= 0x7f800000) {
        sfpi::vFloat product = a * w;
        if constexpr (!is_fp32_dest_acc_en) {
            product = float32_to_bf16_rne(product);
            // multiply follows the FPU for bfloat16: x * 0 = 0. a is nonzero here.
            v_if(w == 0) { product = 0.0f; }
            v_endif;
        }
        a = product;
    }
    v_endif;

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
