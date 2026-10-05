// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_bf16_horner.h"
#include "ckernel_sfpu_bf16_dense_polynomial.h"

namespace ckernel::sfpu::bf16 {
template <typename Config, int Iterations = 32>
inline void calculate_dense_polynomial() {
    static_assert(Iterations == 32, "dense polynomial requires one complete tile");
    static_assert(Config::kRowBase == 64 && Config::kRowLimit == 108 && Config::kOrdered);
    static_assert(Config::kRowBase + Config::kParkedCount <= Config::kRowLimit);
    using Transport = sfpi::dense_polynomial_transport<Config>;
    Transport::template park_boundaries<1>();
    Transport::template park_coeffs<0>();
    for (int d = 0; d < 32; d += 2) {
        sfpi::vFloat x_raw1 = sfpi::dst_reg[d];
        sfpi::vFloat x_raw2 = sfpi::dst_reg[d + 1];
        sfpi::vFloat x1, x2;
        if constexpr (Config::kRawIngress) {
            x1 = sfpi::dense_lower_clamp<Config::kOrdered, Config::kLowerBits>(
                x_raw1, sfpi::dst_reg[d].template mode<sfpi::DataLayout::U16>());
            x2 = sfpi::dense_lower_clamp<Config::kOrdered, Config::kLowerBits>(
                x_raw2, sfpi::dst_reg[d + 1].template mode<sfpi::DataLayout::U16>());
        } else {
            x1 = sfpi::dense_lower_clamp<Config::kOrdered, Config::kLowerBits>(x_raw1);
            x2 = sfpi::dense_lower_clamp<Config::kOrdered, Config::kLowerBits>(x_raw2);
        }
        sfpi::vFloat r1, r2;
        Transport::template eval_seg_dual<0>(x1, x2, r1, r2);
        Transport::template cascade_dual<1>(x1, x2, r1, r2);
        if constexpr (Config::kRawTerminal) {
            sfpi::vUInt raw1 = sfpi::dst_reg[d].template mode<sfpi::DataLayout::U16>();
            sfpi::vUInt raw2 = sfpi::dst_reg[d + 1].template mode<sfpi::DataLayout::U16>();
            r1 = sfpi::dense_lower_clamp<Config::kOrdered, Config::kTerminalBits>(r1);
            sfpi::dense_negative_nan_constant<Config::kTerminalBits>(raw1, r1);
            r2 = sfpi::dense_lower_clamp<Config::kOrdered, Config::kTerminalBits>(r2);
            sfpi::dense_negative_nan_constant<Config::kTerminalBits>(raw2, r2);
        } else {
            r1 = sfpi::dense_lower_clamp<Config::kOrdered, Config::kTerminalBits>(r1);
            r2 = sfpi::dense_lower_clamp<Config::kOrdered, Config::kTerminalBits>(r2);
        }
        r1 = sfpi::convert<sfpi::vFloat16b>(r1, sfpi::RoundMode::Nearest);
        r2 = sfpi::convert<sfpi::vFloat16b>(r2, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[d] = r1;
        sfpi::dst_reg[d + 1] = r2;
    }
}
}  // namespace ckernel::sfpu::bf16
