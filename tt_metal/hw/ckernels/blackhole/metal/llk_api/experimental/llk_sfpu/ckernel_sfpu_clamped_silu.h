// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Shared SFPU activation functions for clamped SiLU (GPT-OSS SwiGLU).
// Used by both the standalone ClampedSilu micro-op and fused activation
// inside DRAMStreamingMatmul.

#pragma once

#include <cstdint>

// Fused activation mode constants for DRAMStreamingMatmul.
static constexpr std::uint32_t FUSED_ACT_NONE = 0;
static constexpr std::uint32_t FUSED_ACT_SILU = 1;
static constexpr std::uint32_t FUSED_ACT_CLAMPED_GATE = 2;
static constexpr std::uint32_t FUSED_ACT_CLAMPED_UP = 3;
static constexpr std::uint32_t FUSED_ACT_GELU = 4;
static constexpr std::uint32_t FUSED_ACT_CLAMP_ONLY = 5;
static constexpr std::uint32_t FUSED_ACT_SITU_GATE = 6;
static constexpr std::uint32_t FUSED_ACT_SCALED_TANH = 7;
static constexpr std::uint32_t FUSED_ACT_SILU_SCALED = 8;

// MATH as well as PACK: MatmulSwiGLU drives these from the math thread.
#if defined(TRISC_PACK) || defined(TRISC_MATH)
#include "ckernel_sfpu_sigmoid.h"

namespace ckernel {
namespace sfpu {

// Gate activation: clamp(x, max=limit) * sigmoid(alpha * clamp(x, max=limit))
template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_clamped_silu_gate(std::uint32_t limit_bits, std::uint32_t alpha_bits) {
    sfpi::vFloat alpha_f = sfpi::as<sfpi::vFloat>(sfpi::vInt(alpha_bits));
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat limit_f = sfpi::as<sfpi::vFloat>(sfpi::vInt(limit_bits));

        x = sfpi::min(x, limit_f);

        sfpi::vFloat sig = _sfpu_sigmoid_<is_fp32_dest_acc_en>(alpha_f * x);
        sfpi::vFloat result = x * sig;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::as<sfpi::vFloat>(sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest));
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// Up activation: clamp(x, -limit, limit) + 1.0
template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_clamped_up(std::uint32_t limit_bits) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat limit_f = sfpi::as<sfpi::vFloat>(sfpi::vInt(limit_bits));
        sfpi::vFloat neg_limit = sfpi::setsgn(limit_f, 1);

        x = sfpi::clamp(x, neg_limit, limit_f);

        sfpi::vFloat result = x + 1.0f;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::as<sfpi::vFloat>(sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest));
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// Clamp-only activation: clamp(x, -limit, limit).
template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_clamped(std::uint32_t limit_bits) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat limit_f = sfpi::as<sfpi::vFloat>(sfpi::vInt(limit_bits));
        sfpi::vFloat neg_limit = sfpi::setsgn(limit_f, 1);

        x = sfpi::clamp(x, neg_limit, limit_f);

        sfpi::vFloat result = x;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::as<sfpi::vFloat>(sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest));
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// Kimi K3 SiTU gate: beta * tanh(x / beta) * sigmoid(x).
template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_situ_gate(std::uint32_t beta_bits, std::uint32_t beta_reciprocal_bits) {
    sfpi::vFloat beta = sfpi::as<sfpi::vFloat>(sfpi::vInt(beta_bits));
    sfpi::vFloat beta_reciprocal = sfpi::as<sfpi::vFloat>(sfpi::vInt(beta_reciprocal_bits));
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat tanh_x = 2.0f * _sfpu_sigmoid_<is_fp32_dest_acc_en>(2.0f * x * beta_reciprocal) - 1.0f;
        sfpi::vFloat result = beta * tanh_x;
        result *= _sfpu_sigmoid_<is_fp32_dest_acc_en>(x);

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::as<sfpi::vFloat>(sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest));
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// Optional Kimi K3 SiTU up transform: beta * tanh(x / beta).
template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_scaled_tanh(std::uint32_t beta_bits, std::uint32_t beta_reciprocal_bits) {
    sfpi::vFloat beta = sfpi::as<sfpi::vFloat>(sfpi::vInt(beta_bits));
    sfpi::vFloat beta_reciprocal = sfpi::as<sfpi::vFloat>(sfpi::vInt(beta_reciprocal_bits));
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat result = 2.0f * _sfpu_sigmoid_<is_fp32_dest_acc_en>(2.0f * x * beta_reciprocal) - 1.0f;
        result *= beta;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::as<sfpi::vFloat>(sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest));
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
#endif  // TRISC_PACK || TRISC_MATH
