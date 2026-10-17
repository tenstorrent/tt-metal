// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"
namespace sfpi {
#include "ckernel_sfpu_bf16_min_max.h"
}
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
}
template <typename Config, int Iterations = 32>
inline void calculate_abs_exp_correction() {
    static_assert(Iterations == 32, "parked correction coefficients require the complete tile");
    sfpi::abs_exp_park_coefficients<Config>();
    auto exp = [](sfpi::vFloat x) {
        constexpr bool split_scale_bias = false;
        {
            sfpi::vFloat multiplier = sfpi::vConstFloatPrgm0;
            return sfpi::correction_exp_leaf<
                Config::kExpDegree,
                false,
                false,
                true,
                Config::kResidualFold,
                true,
                split_scale_bias>(x, multiplier, 127.0f, Config::kExpCoefficients);
        }
    };
    auto reciprocal = [](sfpi::vFloat x) {
        return sfpi::correction_finite_reciprocal(x, [](sfpi::vFloat v) { return sfpu_reciprocal_iter<1>(v); });
    };
#pragma GCC unroll 32
    for (int row = 0; row < Iterations; ++row) {
        sfpi::vFloat raw = sfpi::dst_reg[row];
        sfpi::vFloat result;
        result = sfpi::abs_residual_correction<3, 0, Config>(raw, exp, reciprocal);
        result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[row] = result;
    }
}
}  // namespace ckernel::sfpu::bf16
