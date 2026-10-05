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
#include "ckernel_sfpu_bf16_mirrored_terminals.h"
}  // namespace sfpi
#include "ckernel_sfpu_bf16_finite_reciprocal.h"
#include "ckernel_sfpu_bf16_abs_exp_correction_core.h"
namespace ckernel::sfpu::bf16 {
template <typename Config>
inline void init_abs_exp_correction() {
    sfpi::vConstFloatPrgm0 = Config::kMultiplier;
    if constexpr (Config::kCorrectionStore && !Config::kResidualFold) {
        sfpi::vConstFloatPrgm1 = Config::kExpCoefficients[Config::kExpDegree];
        sfpi::vConstFloatPrgm2 = Config::kExpCoefficients[Config::kExpDegree - 1];
    } else {
        sfpi::vConstFloatPrgm1 =
            Config::kExpCoefficients[Config::kExpDegree] * sfpi::correction_exp_scale(Config::kExpDegree);
        sfpi::vConstFloatPrgm2 =
            Config::kExpCoefficients[Config::kExpDegree - 1] * sfpi::correction_exp_scale(Config::kExpDegree - 1);
    }
    if constexpr (Config::kSquareDecay) {
        sfpu_reciprocal_init<false>();
    }
}
template <typename Config, int Iterations = 32>
inline void calculate_abs_exp_correction() {
    static_assert(Iterations == 32, "parked correction coefficients require the complete tile");
    // Residual composite kernels keep their per-call initializer.
    if constexpr (!Config::kSquareDecay) {
        init_abs_exp_correction<Config>();
    }
    sfpi::abs_exp_park_coefficients<Config>();
    auto exp = [](sfpi::vFloat x) {
        constexpr bool split_scale_bias = false;
        if constexpr (!Config::kSquareDecay && !Config::kSkipInputAbs) {
            x = sfpi::setsgn(x, 0);
        }
        if constexpr (Config::kSquareStore) {
            sfpi::vFloat multiplier = sfpi::dst_reg[64].template mode<sfpi::DataLayout::F32>();
            return sfpi::correction_exp_leaf<Config::kExpDegree, true, true, false, false, true, split_scale_bias>(
                x, multiplier, 127.0f, Config::kExpCoefficients);
        } else if constexpr (Config::kCorrectionStore || Config::kResidualFold) {
            sfpi::vFloat multiplier = sfpi::vConstFloatPrgm0;
            return sfpi::correction_exp_leaf<
                Config::kExpDegree,
                false,
                false,
                true,
                Config::kResidualFold,
                true,
                split_scale_bias>(x, multiplier, 127.0f, Config::kExpCoefficients);
        } else {
            return sfpi::correction_exp_leaf<Config::kExpDegree, false, false, false, false, true, split_scale_bias>(
                x, Config::kMultiplier, 127.0f, Config::kExpCoefficients);
        }
    };
    auto reciprocal = [](sfpi::vFloat x) {
        return sfpi::correction_finite_reciprocal(x, [](sfpi::vFloat v) { return sfpu_reciprocal_iter<1>(v); });
    };
#pragma GCC unroll 32
    for (int row = 0; row < Iterations; ++row) {
        sfpi::vFloat raw = sfpi::dst_reg[row];
        sfpi::vFloat result;
        if constexpr (Config::kSquareDecay) {
            result = sfpi::abs_square_correction<1, 2, Config>(raw, exp, reciprocal);
        } else {
            result = sfpi::abs_residual_correction<3, 0, Config>(raw, exp, reciprocal);
        }
        if constexpr (Config::kDomainActionCount > 0) {
            sfpi::apply_raw_domain_records<Config, Config::kDomainActionCount - 1>(raw, result);
        }
        if constexpr (!Config::kSquareDecay && Config::kRawNanClass) {
            // A NaN of either sign stores +Inf; the affine part would keep the NaN's sign.
            sfpi::nan_class_terminal<1>(sfpi::dst_reg[row].template mode<sfpi::DataLayout::U16>(), result);
        }
        if constexpr (Config::kSquareDecay) {
            sfpi::vUInt encoded = sfpi::dst_reg[row].template mode<sfpi::DataLayout::U16>();
            sfpi::positive_nonfinite_terminal<6>(encoded, result, Config::kPositiveNonfiniteConstant);
            sfpi::negative_nan_class_terminal<1, true>(encoded, result);
        }
        result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[row] = result;
    }
}
}  // namespace ckernel::sfpu::bf16
