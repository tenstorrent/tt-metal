// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_bf16_horner.h"
namespace sfpi {
#include "ckernel_sfpu_bf16_mirrored_terminals.h"
#include "ckernel_sfpu_bf16_symmetric_factored_log_core.h"
}  // namespace sfpi
namespace ckernel::sfpu::bf16 {
template <typename Config, int Iterations = 32>
inline void calculate_symmetric_factored_log() {
    static_assert(Iterations == 32, "selected factored log requires its complete dual-row tile");
    for (int d = 0; d < 32; d += 2) {
        sfpi::vFloat x1 = sfpi::dst_reg[d];
        sfpi::vFloat x2 = sfpi::dst_reg[d + 1];
        v_if(x1 < __builtin_bit_cast(float, Config::kLowerBits)) { x1 = __builtin_bit_cast(float, Config::kLowerBits); }
        v_elseif(x1 > __builtin_bit_cast(float, Config::kUpperBits)) {
            x1 = __builtin_bit_cast(float, Config::kUpperBits);
        }
        v_endif;
        v_if(x2 < __builtin_bit_cast(float, Config::kLowerBits)) { x2 = __builtin_bit_cast(float, Config::kLowerBits); }
        v_elseif(x2 > __builtin_bit_cast(float, Config::kUpperBits)) {
            x2 = __builtin_bit_cast(float, Config::kUpperBits);
        }
        v_endif;
        sfpi::vFloat r1, r2;
        sfpi::symmetric_factored_core_dual<8, Config::kMirrorFold>(
            Config::kNegative, Config::kPositive, 0.0f, x1, x2, r1, r2);
        r1 = r1 * x1;
        r2 = r2 * x2;
        sfpi::symmetric_direct_log_tail<Config::kLowerBits, Config::kUpperBits, Config::kAddendBits>(
            sfpi::dst_reg[d], r1, Config::kLog);
        sfpi::symmetric_direct_log_tail<Config::kLowerBits, Config::kUpperBits, Config::kAddendBits>(
            sfpi::dst_reg[d + 1], r2, Config::kLog);
        sfpi::nan_class_terminal<0>(sfpi::dst_reg[d].template mode<sfpi::DataLayout::U16>(), r1);
        sfpi::nan_class_terminal<0>(sfpi::dst_reg[d + 1].template mode<sfpi::DataLayout::U16>(), r2);
        r1 = sfpi::convert<sfpi::vFloat16b>(r1, sfpi::RoundMode::Nearest);
        r2 = sfpi::convert<sfpi::vFloat16b>(r2, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[d] = r1;
        sfpi::dst_reg[d + 1] = r2;
    }
}
}  // namespace ckernel::sfpu::bf16
