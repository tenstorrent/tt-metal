// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_softplus_bf16_replay_window.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_softplus_bf16_min_max.h"
#include "ckernel_sfpu_softplus_bf16_finite_reciprocal.h"
#include "ckernel_sfpu_softplus_bf16_abs_exp_correction_core.h"
namespace ckernel::sfpu::bf16::softplus {
// The call loads every constant it reads, as stock's reads none its init sets. Wormhole's
// reciprocal still reads its Newton constant from Prgm0.
template <typename Config>
inline void init_abs_exp_correction() {}
template <typename Config, int Iterations = 32>
inline void calculate_abs_exp_correction() {
    static_assert(Iterations == 32, "parked correction coefficients require the complete tile");
    ttpoly::softplus::abs_exp_park_coefficients<Config>();
    auto exp = [](sfpi::vFloat x) __attribute__((always_inline)) {
        constexpr bool split_scale_bias = false;
        {
            sfpi::vFloat multiplier = Config::kMultiplier;
            return ttpoly::softplus::correction_exp_leaf<
                Config::kExpDegree,
                false,
                false,
                true,
                Config::kResidualFold,
                true,
                split_scale_bias,
                sfpi::vFloat,
                true>(x, multiplier, 127.0f, Config::kExpCoefficients);
        }
    };
    auto reciprocal = [](sfpi::vFloat x) __attribute__((always_inline)) {
        auto iteration = [](sfpi::vFloat v) __attribute__((always_inline)) { return sfpu_reciprocal_iter<1>(v); };
        return ttpoly::softplus::correction_finite_reciprocal<decltype(iteration), true>(x, iteration);
    };
#pragma GCC unroll 32
    for (int row = 0; row < Iterations; ++row) {
        sfpi::vFloat raw = sfpi::dst_reg[row];
        sfpi::vFloat result;
        result = ttpoly::softplus::abs_residual_correction<3, 0, Config>(raw, exp, reciprocal);
        result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[row] = result;
    }
}
}  // namespace ckernel::sfpu::bf16::softplus
