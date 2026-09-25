// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#if defined(ARCH_BLACKHOLE)
namespace ckernel::sfpu {
template <bool APPROXIMATION_MODE, int ITERATIONS>
void calculate_softshrink(uint32_t param0);
}
#endif
namespace sfpi {
#include "ckernel_sfpu_tt_poly_min_max.h"
#include "ckernel_sfpu_tt_poly_simple_algebraic.h"
}  // namespace sfpi
namespace ckernel::sfpu::ttpoly {
template <typename Config>
inline void init_simple_forward() {
    if constexpr (Config::kRowsPerReplay) {
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2 * Config::kRowsPerReplay}}.set(
            ADDR_MOD_6);
    }
}
template <typename Config, int Iterations = 8>
inline void calculate_simple_forward() {
    static_assert(Iterations > 0 && Iterations % 2 == 0);
    if constexpr (Config::kBodySlots) {
        if constexpr (Config::kKind == 1) {
            constexpr uint32_t pin = (Config::kComparatorBf16 << 16) ^ 0x80000000u;
            TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, pin >> 16);
            TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, pin & 65535);
        } else {
            TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER, Config::kSlopeBits >> 16);
            TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, Config::kSlopeBits & 65535);
#if defined(ARCH_WORMHOLE)
            TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_USHORT, Config::kRawEqual);
#else
            TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, Config::kInterceptBits >> 16);
            TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, Config::kInterceptBits & 65535);
#endif
        }
        TTI_REPLAY(0, Config::kBodySlots, 1, 1);
#if defined(ARCH_BLACKHOLE)
        if constexpr (Config::kKind == 1) {
            sfpi::simple_threshold_pair<ADDR_MOD_7, ADDR_MOD_6>();
        } else if constexpr (Config::kRowsPerReplay == 2) {
            sfpi::simple_gated_pair<false, true, Config::kRawEqual, 0, 0, ADDR_MOD_7, ADDR_MOD_6>();
        } else {
            sfpi::simple_bh_gated_joint_row<Config::kRawEqual, ADDR_MOD_7, ADDR_MOD_6>();
        }
#elif defined(ARCH_WORMHOLE)
        sfpi::simple_wh_gated_row<0, ADDR_MOD_3, ADDR_MOD_2>();
#else
        static_assert(sizeof(Config) == 0, "simple forward requires BH/WH");
#endif
#pragma GCC unroll 8
        for (int row = Config::kRowsPerReplay; row < Iterations; row += Config::kRowsPerReplay) {
            TTI_REPLAY(0, Config::kBodySlots, 0, 0);
        }
    } else {
#if defined(ARCH_BLACKHOLE)
        static_assert(Config::kKind == 2);
        // This is the selected canonical native primitive, not a new evaluator.
        ckernel::sfpu::calculate_softshrink<false, Iterations>(Config::kThresholdBits);
#elif defined(ARCH_WORMHOLE)
#pragma GCC unroll 8
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
#else
        static_assert(sizeof(Config) == 0, "simple forward requires BH/WH");
#endif
    }
}
}  // namespace ckernel::sfpu::ttpoly
#define TT_POLY_LLK_SIMPLE_FORWARD_SELECTED_V1 1
#if defined(ARCH_BLACKHOLE)
// The stock wrapper also imports its generated configuration in composed
// packages. Expose this runtime before entering that wrapper's include cycle.
#include "ckernel_sfpu_softshrink.h"
#endif
