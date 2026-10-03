// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
namespace sfpi {
#include "ckernel_sfpu_bf16_min_max.h"
#include "ckernel_sfpu_bf16_simple_algebraic.h"
}  // namespace sfpi
namespace ckernel::sfpu::bf16 {
template <typename Config>
inline void init_simple_forward() {
    // Only the replay's advance mode needs programming. A threshold pair that
    // folds its identity store (always on WH, at 11 slots on BH) advances with
    // INCRWC and never reads it.
    constexpr bool folds = Config::kKind == 1;
    if constexpr (Config::kRowsPerReplay && !folds) {
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2 * Config::kRowsPerReplay}}.set(
            ADDR_MOD_6);
    }
}
template <typename Config, int Iterations = 8>
inline void calculate_simple_forward() {
    static_assert(Iterations > 0 && Iterations % 2 == 0);
    if constexpr (Config::kBodySlots) {
        if constexpr (Config::kKind == 1) {
            constexpr uint32_t pin =
                Config::kBodySlots != 16 ? Config::kThresholdBits : (Config::kComparatorBf16 << 16) ^ 0x80000000u;
            TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, pin >> 16);
            TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, pin & 65535);
        } else {
            TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER, Config::kSlopeBits >> 16);
            TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, Config::kSlopeBits & 65535);
            TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_USHORT, Config::kRawEqual);
        }
        TTI_REPLAY(0, Config::kBodySlots, 1, 1);
        if constexpr (Config::kKind == 1) {
            static_assert(Config::kBodySlots == 13 && Config::kRowsPerReplay == 2);
            sfpi::simple_threshold_pair<ADDR_MOD_3, ADDR_MOD_2, true>();
        } else {
            sfpi::simple_wh_gated_row<0, ADDR_MOD_3, ADDR_MOD_2>();
        }
#pragma GCC unroll 8
        for (int row = Config::kRowsPerReplay; row < Iterations; row += Config::kRowsPerReplay) {
            TTI_REPLAY(0, Config::kBodySlots, 0, 0);
        }
    } else {
#pragma GCC unroll 32
        for (int row = 0; row < Iterations; ++row) {
            sfpi::vFloat x = sfpi::dst_reg[0];
            sfpi::vFloat y;
            if constexpr (Config::kKind == 1) {
                y = sfpi::simple_threshold_identity(x, __builtin_bit_cast(float, Config::kThresholdBits));
            } else if constexpr (Config::kKind == 2) {
                y = sfpi::simple_threshold_softshift(x, __builtin_bit_cast(float, Config::kThresholdBits));
            } else {
                y = sfpi::simple_gated_product(
                    x,
                    __builtin_bit_cast(float, Config::kInterceptBits),
                    __builtin_bit_cast(float, Config::kSlopeBits));
            }
            y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
            sfpi::dst_reg[0] = y;
            sfpi::dst_reg++;
        }
    }
}
}  // namespace ckernel::sfpu::bf16
