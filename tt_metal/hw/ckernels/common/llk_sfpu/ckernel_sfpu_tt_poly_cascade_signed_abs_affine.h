// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_tt_poly_horner.h"
#include "ckernel_sfpu_tt_poly_signed_abs_affine.h"

namespace ckernel::sfpu::ttpoly {
template <typename Config>
struct affine_coefficients {
    constexpr float operator[](uint32_t index) const {
        return __builtin_bit_cast(float, Config::kCoefficientBits[index]);
    }
};

template <typename Config>
inline void init_signed_abs_affine() {
#if defined(ARCH_BLACKHOLE)
    sfpi::signed_abs_affine_init<Config>();
#elif !defined(ARCH_WORMHOLE)
    static_assert(sizeof(Config) == 0, "signed-abs affine requires BH or WH");
#endif
}

template <typename Config, int Iterations = 8>
inline void calculate_signed_abs_affine() {
    static_assert(Iterations > 0 && Iterations % 2 == 0);
#if defined(ARCH_BLACKHOLE)
    sfpi::signed_abs_affine_pin_coefficients<Config>();
    TTI_REPLAY(0, Config::kBodySlots, 1, 1);
    sfpi::signed_abs_affine_body<Config, ADDR_MOD_7, ADDR_MOD_6>();
#pragma GCC unroll 8
    for (int row = 1; row < Iterations; ++row) {
        TTI_REPLAY(0, Config::kBodySlots, 0, 0);
    }
    sfpi::signed_abs_affine_drain();
    // No deployment SETRWC: the upstream LLK owns face traversal/counters.
#elif defined(ARCH_WORMHOLE)
#pragma GCC unroll 4
    for (int row = 0; row < Iterations / 2; ++row) {
        sfpi::vFloat raw0 = sfpi::dst_reg[0];
        sfpi::vFloat raw1 = sfpi::dst_reg[1];
        sfpi::vFloat x0 = sfpi::setsgn(raw0, 0);
        sfpi::vFloat x1 = sfpi::setsgn(raw1, 0);
        sfpi::vFloat y0, y1;
        sfpi::eval_polynomial_dual<Config::kDegree>(affine_coefficients<Config>{}, x0, x1, y0, y1);
        sfpi::signed_abs_affine_tail<Config>(x0, y0);
        sfpi::signed_abs_affine_tail<Config>(x1, y1);
        y0 = sfpi::copysgn(y0, raw0);
        y1 = sfpi::copysgn(y1, raw1);
        y0 = sfpi::convert<sfpi::vFloat16b>(y0, sfpi::RoundMode::Nearest);
        y1 = sfpi::convert<sfpi::vFloat16b>(y1, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[0] = y0;
        sfpi::dst_reg[1] = y1;
        sfpi::dst_reg += 2;
    }
#else
    static_assert(sizeof(Config) == 0, "signed-abs affine requires BH or WH");
#endif
}
}  // namespace ckernel::sfpu::ttpoly
#define TT_POLY_LLK_SIGNED_ABS_AFFINE_SELECTED_V1 1
