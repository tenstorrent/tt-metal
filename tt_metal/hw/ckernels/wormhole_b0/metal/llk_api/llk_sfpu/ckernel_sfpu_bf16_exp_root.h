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
template <typename Config>
inline void init_exp_root() {
    sfpi::vConstIntPrgm0 = Config::kRootMagic;
    sfpi::vConstFloatPrgm1 = Config::kRootC1;
    sfpi::vConstFloatPrgm2 = Config::kRootC2;
}

template <typename Config, int Iterations = 32>
inline void calculate_exp_root() {
    static_assert(Iterations == 32, "selected exp-root requires a complete tile");
    static_assert(
        Config::kDomainActionCount == 2 && (Config::kLateNegativeNanClass == 1 || Config::kLateNegativeNanClass == 2));
    static_assert(Config::kWormhole);
    for (int d = 0; d < 32; d++) {
        sfpi::vFloat x_raw = sfpi::dst_reg[d];
        sfpi::vFloat y = sfpi::exp_root_eval<Config>(x_raw);
        sfpi::apply_raw_domain_records<Config, 1>(x_raw, y);
        sfpi::vUInt raw_u16 = sfpi::dst_reg[d].template mode<sfpi::DataLayout::U16>();
        {
            // -Inf and every -NaN share the class: sign and all-ones exponent, any mantissa.
            sfpi::vUInt sign_and_exponent_delta = (raw_u16 ^ sfpi::vUInt(0x80ffu)) & sfpi::vUInt(0x80ffu);
            v_if(sign_and_exponent_delta == 0u) {
                y = sfpi::target_raw_terminal_value<Config::kLateNegativeNanClass>(y);
            }
            v_endif;
        }
        y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[d] = y;
    }
}
}  // namespace ckernel::sfpu::bf16
