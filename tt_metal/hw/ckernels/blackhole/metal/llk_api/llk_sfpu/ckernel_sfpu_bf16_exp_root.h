// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
namespace sfpi {
#include "ckernel_sfpu_bf16_mirrored_terminals.h"
#include "ckernel_sfpu_bf16_exp_root_core.h"
}  // namespace sfpi
namespace ckernel::sfpu::bf16 {
template <typename Config, int Iterations = 32>
inline void calculate_exp_root() {
    static_assert(Iterations == 32, "selected exp-root requires a complete tile");
    static_assert(
        Config::kDomainActionCount == 2 && (Config::kLateNegativeNanClass == 1 || Config::kLateNegativeNanClass == 2));
    static_assert(!Config::kWormhole);
    sfpi::vConstIntPrgm0 = Config::kRootMagic;
    sfpi::vConstFloatPrgm1 = Config::kRootC1;
    sfpi::vConstFloatPrgm2 = Config::kRootC2;
    for (int d = 0; d < 32; d++) {
        sfpi::vFloat x_raw = sfpi::dst_reg[d];
        sfpi::vFloat y = sfpi::exp_root_eval<Config>(x_raw);
        sfpi::apply_raw_domain_records<Config, 1>(x_raw, y);
        sfpi::vUInt raw_u16 = sfpi::dst_reg[d].template mode<sfpi::DataLayout::U16>();
        sfpi::negative_nan_class_terminal<Config::kLateNegativeNanClass, true>(raw_u16, y);
        y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[d] = y;
    }
}
}  // namespace ckernel::sfpu::bf16
