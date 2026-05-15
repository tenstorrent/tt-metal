// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

// Verified degree-5 minimax coefficients for ln(m), m ∈ [1,2)
constexpr float LOG_A =  0.0304490048f;  // x^5
constexpr float LOG_B = -0.2849152297f;  // x^4
constexpr float LOG_C =  1.1226603286f;  // x^3
constexpr float LOG_D = -2.4546323801f;  // x^2
constexpr float LOG_E =  3.5282313183f;  // x^1
constexpr float LOG_F = -1.9417930419f;  // constant

template <bool HAS_BASE_SCALING>
sfpi_inline void _calculate_log_body_(const std::uint32_t log_base_scale_factor, const std::uint32_t dst_idx = 0)
{
    constexpr std::uint32_t dst_tile_size_sfpi = 32;

    sfpi::vFloat in = sfpi::dst_reg[dst_idx * dst_tile_size_sfpi];
    sfpi::vFloat x = setexp(in, 127);

    // Degree-5 Horner (verified minimax)
    sfpi::vFloat series = x * (x * (x * (x * (x * LOG_A + LOG_B) + LOG_C) + LOG_D) + LOG_E) + LOG_F;

    sfpi::vInt exp = exexp(in);
    v_if (exp < 0) {
        exp = sfpi::setsgn(~exp + 1, 1);
    }
    v_endif;
    sfpi::vFloat expf = int32_to_float(exp, sfpi::RoundMode::NearestEven);

    sfpi::vFloat ln2 = sfpi::vConstFloatPrgm0;   // ln(2)
    sfpi::vFloat result = expf * ln2 + series;

    if constexpr (HAS_BASE_SCALING) {
        result *= sfpi::sFloat16a(log_base_scale_factor);
    }

    v_if (in == 0.0F) {
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    sfpi::dst_reg[dst_idx * dst_tile_size_sfpi] = result;
}

sfpi_inline sfpi::vFloat _calculate_log_body_no_init_(sfpi::vFloat base)
{
    sfpi::vFloat x = setexp(base, 127);

    sfpi::vFloat series = x * (x * (x * (x * (x * LOG_A + LOG_B) + LOG_C) + LOG_D) + LOG_E) + LOG_F;

    sfpi::vInt exp = exexp(base);
    v_if (exp < 0) {
        exp = sfpi::setsgn(~exp + 1, 1);
    }
    v_endif;
    sfpi::vFloat expf = int32_to_float(exp, sfpi::RoundMode::NearestEven);

    sfpi::vFloat ln2 = 0.692871f;
    sfpi::vFloat result = expf * ln2 + series;

    v_if (base == 0.0f) {
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    return result;
}

// The rest of the file (_calculate_log_, _init_log_) remains unchanged
template <bool APPROXIMATION_MODE, bool HAS_BASE_SCALING, int ITERATIONS>
inline void _calculate_log_(const int iterations, std::uint32_t log_base_scale_factor)
{
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++) {
        _calculate_log_body_<HAS_BASE_SCALING>(log_base_scale_factor);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void _init_log_()
{
    sfpi::vConstFloatPrgm0 = 0.692871f;   // ln(2)
}

} // namespace sfpu
} // namespace ckernel
