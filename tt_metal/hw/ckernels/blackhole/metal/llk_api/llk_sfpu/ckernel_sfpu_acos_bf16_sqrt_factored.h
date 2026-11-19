// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_acos_bf16_horner.h"
#include "ckernel_sfpu_acos_bf16_sqrt_factored_core.h"
#include "ckernel_sfpu_acos_bf16_mirrored_terminals.h"

namespace ckernel::sfpu::bf16::acos {
template <typename Config, int Iterations = 32>
inline void calculate_sqrt_factored() {
    static_assert(Iterations == 32, "reflected root requires the selected complete-tile row loop");
    static_assert(!Config::kSafeInput);
    for (int d = 0; d < 32; ++d) {
        sfpi::vFloat raw = sfpi::dst_reg[d];
        sfpi::vFloat input = raw;
        sfpi::vFloat magnitude = sfpi::setsgn(input, 0);
        sfpi::vFloat result = ttpoly::acos::eval_polynomial<Config::kDegree>(Config{}, magnitude);
        result = ttpoly::acos::sqrt_factored_product(result, magnitude);
        result = ttpoly::acos::sqrt_factored_reflect(input, result);
        ttpoly::acos::mirrored_class_terminals<0x3f800000u, false>(raw, result);
        result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[d] = result;
    }
}
}  // namespace ckernel::sfpu::bf16::acos
