// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_bf16_horner.h"
namespace sfpi {
#include "ckernel_sfpu_bf16_sqrt_factored_core.h"
#include "ckernel_sfpu_bf16_mirrored_terminals.h"
}  // namespace sfpi

namespace ckernel::sfpu::bf16 {
template <typename Config, int Iterations = 32>
inline void calculate_sqrt_factored() {
    static_assert(Iterations == 32, "reflected root requires the selected complete-tile row loop");
    static_assert(!Config::kSafeInput);
    for (int d = 0; d < 32; ++d) {
        sfpi::vFloat raw = sfpi::dst_reg[d];
        sfpi::vFloat input = raw;
        if constexpr (Config::kSafeInput) {
            input = sfpi::max(input, -1.0f);
            input = sfpi::min(input, 1.0f);
        }
        sfpi::vFloat magnitude = sfpi::setsgn(input, 0);
        sfpi::vFloat result = sfpi::eval_polynomial<Config::kDegree>(Config{}, magnitude);
        result = sfpi::sqrt_factored_product(result, magnitude);
        result = sfpi::sqrt_factored_reflect(input, result);
        sfpi::mirrored_class_terminals<0x3f800000u, false>(raw, result);
        result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[d] = result;
    }
}
}  // namespace ckernel::sfpu::bf16
