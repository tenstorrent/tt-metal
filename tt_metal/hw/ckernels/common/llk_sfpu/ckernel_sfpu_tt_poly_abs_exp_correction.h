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
#include "ckernel_sfpu_tt_poly_min_max.h"
#include "ckernel_sfpu_tt_poly_mirrored_terminals.h"
}  // namespace sfpi
#include "ckernel_sfpu_tt_poly_finite_reciprocal.h"
#include "ckernel_sfpu_tt_poly_abs_exp_correction_core.h"
namespace ckernel::sfpu::ttpoly {
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
    // Residual composite kernels retain their existing per-call initializer.
    if constexpr (!Config::kSquareDecay) {
        init_abs_exp_correction<Config>();
    }
#if defined(ARCH_BLACKHOLE)
    if constexpr (!Config::kSquareDecay) {
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
        sfpi::abs_exp_residual_pins<Config>();
        constexpr uint32_t slots = 27u + (Config::kNumerator[0] == 1.0f ? 0u : 1u);
        TTI_REPLAY(0, slots, 1, 1);
        sfpi::abs_exp_residual_core<Config, ADDR_MOD_7>();
        sfpi::abs_exp_residual_suffix<Config, ADDR_MOD_6>();
#pragma GCC unroll 32
        for (int row = 1; row < Iterations; ++row) {
            TTI_REPLAY(0, slots, 0, 0);
            sfpi::abs_exp_residual_suffix<Config, ADDR_MOD_6>();
        }
        return;
    }
#elif !defined(ARCH_WORMHOLE)
#error "selected abs-exp correction requires BH or WH"
#endif
    sfpi::abs_exp_park_coefficients<Config>();
    auto exp = [](sfpi::vFloat x) {
        if constexpr (!Config::kSquareDecay && !Config::kSkipInputAbs) {
            x = sfpi::setsgn(x, 0);
        }
        if constexpr (Config::kSquareStore) {
            sfpi::vFloat multiplier = sfpi::dst_reg[64].template mode<sfpi::DataLayout::F32>();
            return sfpi::correction_exp_leaf<Config::kExpDegree, true, true, false, false, true>(
                x, multiplier, 127.0f, Config::kExpCoefficients);
        } else if constexpr (Config::kCorrectionStore) {
            sfpi::vFloat multiplier = sfpi::vConstFloatPrgm0;
            return sfpi::correction_exp_leaf<Config::kExpDegree, false, false, true, Config::kResidualFold, true>(
                x, multiplier, 127.0f, Config::kExpCoefficients);
        } else {
            return sfpi::correction_exp_leaf<Config::kExpDegree, false, false, false, false, true>(
                x, Config::kMultiplier, 127.0f, Config::kExpCoefficients);
        }
    };
    auto reciprocal = [](sfpi::vFloat x) {
        return sfpi::correction_finite_reciprocal(x, [](sfpi::vFloat v) { return sfpu_reciprocal_iter<1>(v); });
    };
#if defined(ARCH_WORMHOLE)
#pragma GCC unroll 32
#endif
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
        if constexpr (Config::kSquareDecay) {
            sfpi::vUInt encoded = sfpi::dst_reg[row].template mode<sfpi::DataLayout::U16>();
#if defined(ARCH_BLACKHOLE)
            sfpi::signed_nonfinite_split_terminal<6>(encoded, result, Config::kPositiveNonfiniteConstant);
#else
            sfpi::positive_nonfinite_terminal<6>(encoded, result, Config::kPositiveNonfiniteConstant);
            sfpi::negative_nan_class_terminal<1, true>(encoded, result);
#endif
        }
        result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[row] = result;
    }
}
}  // namespace ckernel::sfpu::ttpoly
#define TT_POLY_LLK_ABS_EXP_CORRECTION_SELECTED_V1 1
