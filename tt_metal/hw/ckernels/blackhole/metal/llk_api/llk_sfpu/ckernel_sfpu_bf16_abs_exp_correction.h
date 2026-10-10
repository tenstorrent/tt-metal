// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ckernel_sfpu_bf16_sfpi_isa.h"
#include <cstdint>
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"
namespace sfpi {
#include "ckernel_sfpu_bf16_min_max.h"
#include "ckernel_sfpu_bf16_mirrored_terminals.h"
}  // namespace sfpi
#include "ckernel_sfpu_bf16_finite_reciprocal.h"
#include "ckernel_sfpu_bf16_abs_exp_correction_core.h"
namespace ckernel::sfpu::bf16 {
template <typename Config>
inline void init_abs_exp_correction() {
    sfpi::vConstFloatPrgm0 = Config::kMultiplier;
    sfpi::vConstFloatPrgm1 =
        Config::kExpCoefficients[Config::kExpDegree] * sfpi::correction_exp_scale(Config::kExpDegree);
    sfpi::vConstFloatPrgm2 =
        Config::kExpCoefficients[Config::kExpDegree - 1] * sfpi::correction_exp_scale(Config::kExpDegree - 1);
    sfpu_reciprocal_init<false>();
}
template <typename Config, int Iterations = 32>
inline void calculate_abs_exp_correction() {
    static_assert(Iterations == 32, "parked correction coefficients require the complete tile");
    // Residual composite kernels keep their per-call initializer.
    sfpi::abs_exp_park_coefficients<Config>();
    auto exp = [](sfpi::vFloat x) {
        constexpr bool split_scale_bias = !Config::kSquareDecay;
        return sfpi::correction_exp_leaf<Config::kExpDegree, false, false, false, false, true, split_scale_bias>(
            x, Config::kMultiplier, 127.0f, Config::kExpCoefficients);
    };
    auto reciprocal = [](sfpi::vFloat x) {
        return sfpi::correction_finite_reciprocal(x, [](sfpi::vFloat v) { return sfpu_reciprocal_iter<1>(v); });
    };
    for (int row = 0; row < Iterations; ++row) {
        sfpi::vFloat raw = sfpi::dst_reg[row];
        sfpi::vFloat result;
        result = sfpi::abs_square_correction<1, 2, Config>(raw, exp, reciprocal);
        sfpi::vUInt encoded = sfpi::dst_reg[row].template mode<sfpi::DataLayout::U16>();
        sfpi::signed_nonfinite_split_terminal<6>(encoded, result, Config::kPositiveNonfiniteConstant);
        result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[row] = result;
    }
}
}  // namespace ckernel::sfpu::bf16
