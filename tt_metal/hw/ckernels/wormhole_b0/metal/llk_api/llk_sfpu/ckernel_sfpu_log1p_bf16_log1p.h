// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_log1p_bf16_replay_window.h"
#include "ckernel_sfpu_log1p_bf16_log1p_replay.h"
namespace ckernel::sfpu::bf16::log1p {
constexpr uint32_t kLog1pHold = ADDR_MOD_3;

template <typename Config, int Iterations = 8>
inline void calculate_log1p() {
    static_assert(Iterations == 8 || Iterations == 32);
    static_assert(Config::kBodySlots == 31u);
    static_assert(Config::kLowerBits == 0x3f400000u && Config::kQuotientBits == 0x7f800000u);
    static_assert(Config::kRawEqual == 0xffu);
    constexpr uint32_t c0 = Config::kCoefficientBits[0];
    // WH reads c1 and c2 from Prgm1 and Prgm2, which stock log1p_init owns: this call writes them and
    // puts back that init's values.
    constexpr uint32_t c1 = Config::kCoefficientBits[1], c2 = Config::kCoefficientBits[2];
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(float, c2);
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, c1);
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, c0 >> 16);
    TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, c0 & 0xffffu);
    auto body = [&] {
        ttpoly::log1p::
            log1p_replay_body<2, true, kLog1pHold, Config::kLowerBits, Config::kQuotientBits, Config::kRawEqual>();
    };
    record_row<0, Config::kBodySlots>(body);
    TTI_SFPENCC(0, 0, 0, 0);
    TTI_SFPSTORE(p_sfpu::LREG0, 0, kLog1pHold, 0);
    TTI_INCRWC(0, 2, 0, 0);
#pragma GCC unroll 8
    for (int row = 1; row < Iterations; ++row) {
        replay_row<0, Config::kBodySlots>(body);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, 0, kLog1pHold, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(float, Config::kSharedPrgm1Bits);
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, Config::kSharedPrgm2Bits);
}
}  // namespace ckernel::sfpu::bf16::log1p
