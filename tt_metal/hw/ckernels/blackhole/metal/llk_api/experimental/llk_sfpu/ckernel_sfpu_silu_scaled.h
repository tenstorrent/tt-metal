// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Scaled SiLU: dst = tail * silu(s*x), for broadcast scalars s and tail.
//
// SiLU is not scale-invariant, so a scalar belonging on the activation's INPUT
// cannot be applied afterwards; s goes inside, tail outside.
//
// Requires the caller's silu_init: this shares calculate_silu's non-approx sigmoid.
// Not TRISC_PACK-gated — callers invoke it from both MATH and PACK.

#pragma once

#include <cstdint>

#if defined(COMPILE_FOR_TRISC)

#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_sigmoid.h"  // _sfpu_sigmoid_
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// Clear HAS_TAIL_SCALE when a downstream op already applies s.
template <bool is_fp32_dest_acc_en, bool HAS_TAIL_SCALE, bool HAS_POST_SCALE, int ITERATIONS>
inline void calculate_silu_scaled(std::uint32_t scale_bits, std::uint32_t post_scale_bits) {
    constexpr bool HAS_TAIL = HAS_TAIL_SCALE || HAS_POST_SCALE;
    sfpi::vFloat s = sfpi::as<sfpi::vFloat>(sfpi::vInt(scale_bits));
    sfpi::vFloat tail = s;
    if constexpr (HAS_TAIL) {
        if constexpr (HAS_TAIL_SCALE && HAS_POST_SCALE) {
            tail = s * sfpi::as<sfpi::vFloat>(sfpi::vInt(post_scale_bits));
        } else if constexpr (HAS_POST_SCALE) {
            tail = sfpi::as<sfpi::vFloat>(sfpi::vInt(post_scale_bits));
        }
    }
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0] * s;

        sfpi::vFloat result = x * _sfpu_sigmoid_<is_fp32_dest_acc_en>(x);
        if constexpr (HAS_TAIL) {
            result = result * tail;
        }

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu

// Template order mirrors llk_math_eltwise_unary_sfpu_silu.
template <
    bool APPROXIMATE,
    bool is_fp32_dest_acc_en,
    bool HAS_TAIL_SCALE = true,
    bool HAS_POST_SCALE = false,
    int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_silu_scaled(
    std::uint32_t dst_index,
    std::uint32_t scale_bits,
    std::uint32_t post_scale_bits = 0,
    VectorMode vector_mode = VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::calculate_silu_scaled<is_fp32_dest_acc_en, HAS_TAIL_SCALE, HAS_POST_SCALE, ITERATIONS>,
        dst_index,
        vector_mode,
        scale_bits,
        post_scale_bits);
}

}  // namespace ckernel

#endif  // COMPILE_FOR_TRISC
