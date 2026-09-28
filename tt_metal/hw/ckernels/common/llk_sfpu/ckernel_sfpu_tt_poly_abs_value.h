// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu::ttpoly {
#if defined(ARCH_WORMHOLE)
constexpr uint32_t kAbsHold = ADDR_MOD_3;
constexpr uint32_t kAbsAdvance = ADDR_MOD_2;
constexpr uint32_t kSelectedAbsSlots = 25;
#elif defined(ARCH_BLACKHOLE)
constexpr uint32_t kAbsHold = ADDR_MOD_7;
constexpr uint32_t kAbsAdvance = ADDR_MOD_6;
constexpr uint32_t kSelectedAbsSlots = 23;
#else
#error "absolute-value replay requires BH or WH"
#endif

template <typename Config>
constexpr bool abs_value_contract() {
    return Config::kEqualMask == 0x80ffu && Config::kEqualValue == 0x80ffu && Config::kNonzeroMask == 0x7f00u &&
           Config::kOutputBf16 == 0xff80u && Config::kTruePatterns == 127u &&
           Config::kSelectedBodySlots == kSelectedAbsSlots && Config::kSelectedPeakLregs == 8u &&
           Config::kBodySlots == 6u && Config::kPeakLregs == 2u && Config::kLoadFormat == sfpi::SFPLOAD_MOD0_FMT_SRCB &&
           Config::kAbsMode == sfpi::SFPABS_MOD1_FLOAT && Config::kStoreFormat == sfpi::SFPSTORE_MOD0_FMT_SRCB &&
           Config::kProvedPatterns == 65536u;
}

template <typename Config>
inline void init_abs_value() {
    static_assert(abs_value_contract<Config>());
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_6);
}

template <typename Config, int Iterations = 8>
inline void calculate_abs_value() {
    static_assert(abs_value_contract<Config>());
    static_assert(Iterations > 0 && Iterations % 2 == 0);
    // The typed exhaustive proof absorbs the encoded terminal into the native
    // load/FLOAT-SFPABS/store path. The upstream wrapper owns face traversal.
    TTI_REPLAY(0, Config::kBodySlots, 1, 1);
    TTI_SFPLOAD(p_sfpu::LREG0, Config::kLoadFormat, kAbsHold, 0);
    TTI_SFPLOAD(p_sfpu::LREG1, Config::kLoadFormat, kAbsHold, 2);
    TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG0, Config::kAbsMode);
    TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG1, Config::kAbsMode);
    TTI_SFPSTORE(p_sfpu::LREG0, Config::kStoreFormat, kAbsHold, 0);
    TTI_SFPSTORE(p_sfpu::LREG1, Config::kStoreFormat, kAbsAdvance, 2);
#pragma GCC unroll 4
    for (int pair = 1; pair < Iterations / 2; ++pair) {
        TTI_REPLAY(0, Config::kBodySlots, 0, 0);
    }
}
}  // namespace ckernel::sfpu::ttpoly
#define TT_POLY_LLK_ABS_VALUE_NATIVE_TERMINAL_V2 1
