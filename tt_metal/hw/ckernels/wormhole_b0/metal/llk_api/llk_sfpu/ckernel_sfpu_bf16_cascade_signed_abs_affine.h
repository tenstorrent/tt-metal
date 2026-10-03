// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_bf16_horner.h"
#include "ckernel_sfpu_bf16_signed_abs_affine.h"

namespace ckernel::sfpu::bf16 {
template <typename Config>
struct affine_coefficients {
    constexpr float operator[](uint32_t index) const {
        return __builtin_bit_cast(float, Config::kCoefficientBits[index]);
    }
};

template <typename Config>
inline void init_signed_abs_affine() {
    if constexpr (Config::kNegativeNanTerminal) {
        sfpi::vConstIntPrgm0 = static_cast<int32_t>(0x807f0000u);
    }
}

// The copysign source: raw's sign, but positive for a NaN of either sign. A DEST
// carrier's low half is zero, so |raw| + 0x807f0000 (Prgm0, set in init) is
// negative exactly when raw is not a NaN.
sfpi_inline sfpi::vFloat nan_positive_sign(sfpi::vFloat raw, sfpi::vFloat magnitude) {
    sfpi::vInt not_nan = sfpi::as<sfpi::vInt>(magnitude) + sfpi::vConstIntPrgm0;
    return sfpi::as<sfpi::vFloat>(not_nan & sfpi::as<sfpi::vInt>(raw));
}

template <typename Config, int Iterations = 8>
inline void calculate_signed_abs_affine() {
    static_assert(Iterations > 0 && Iterations % 2 == 0);
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
        if constexpr (Config::kNegativeNanTerminal) {
            y0 = sfpi::copysgn(y0, nan_positive_sign(raw0, x0));
            y1 = sfpi::copysgn(y1, nan_positive_sign(raw1, x1));
        } else {
            y0 = sfpi::copysgn(y0, raw0);
            y1 = sfpi::copysgn(y1, raw1);
        }
        y0 = sfpi::convert<sfpi::vFloat16b>(y0, sfpi::RoundMode::Nearest);
        y1 = sfpi::convert<sfpi::vFloat16b>(y1, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[0] = y0;
        sfpi::dst_reg[1] = y1;
        sfpi::dst_reg += 2;
    }
}
}  // namespace ckernel::sfpu::bf16
