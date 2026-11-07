// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_sqrt_bf16_replay_window.h"
// WH cbrt runs stock's kernel, defined in ckernel_sfpu_cbrt.h, which includes this file.
namespace ckernel::sfpu {
template <bool, bool, int>
inline void calculate_cube_root();
}
namespace ckernel::sfpu::bf16::sqrt {
constexpr uint32_t kNewtonSqrtCompactSlots = 26u;

inline void newton_sqrt_wh_compact_row_head() {
    TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_3, 0);                                  // x
    TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);                                // WH shifts its destination in place
    TTI_SFPSHFT(0xFFF, 0, p_sfpu::LREG0, 1);                                       // bits(x) >> 1
    TTI_SFPIADD(0, p_sfpu::LREG12, p_sfpu::LREG0, 6);                              // y = MAGIC - i
    TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);  // xy
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG3, 1);                                // -y
    TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);  // c = -y*xy
    TTI_SFPLOAD(p_sfpu::LREG7, sfpi::SFPLOAD_MOD0_FMT_UINT16, ADDR_MOD_3, 0);      // raw word
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG14, p_sfpu::LREG1, p_sfpu::LREG3, 0);  // C2 + c
    TTI_SFPABS(0, p_sfpu::LREG2, p_sfpu::LREG2, 1);                                 // |x|
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LREG13, p_sfpu::LREG1, 0);     // t = c*(C2+c) + C1
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_USHORT, 0x8000);
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);  // y *= t
    TTI_SFPLOADI(p_sfpu::LREG6, 0, 0x7F80);                                        // +Inf
    TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);  // xy
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG3, 1);                                // -y
}

inline void newton_sqrt_wh_compact_row_tail() {
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG4, 0);  // 1 - y*xy
    TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG6, 2);                               // |x| < +Inf
    TTI_SFPDIVP2(0xFFF, p_sfpu::LREG4, p_sfpu::LREG3, 1);                          // (1 - y*xy)/2
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG1, p_sfpu::LREG1, p_sfpu::LREG0, 0);     // y = (1-y*xy)/2*xy + xy
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG7, 2);  // 0x8000 - raw < 0: negative, not -0
    TTI_SFPLOADI(p_sfpu::LREG0, 0, 0x7FC0);
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN,
        0,
        p_sfpu::LREG0,
        p_sfpu::LREG0,
        p_sfpu::LREG0,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);  // fp32 -> bf16 RNE
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);
}

template <typename Config, int Iterations>
inline void calculate_newton_root() {
    static_assert(Config::kKind == 0);
    static_assert(Iterations == 8 || Iterations == 32);
    static_assert(Config::kBodySlots == kNewtonSqrtCompactSlots);
    {
        auto head = [&] { newton_sqrt_wh_compact_row_head(); };
        auto tail = [&] { newton_sqrt_wh_compact_row_tail(); };
        record_split<0, Config::kBodySlots, 16>(head, tail);
        TTI_INCRWC(0, 2, 0, 0);
#pragma GCC unroll 8
        for (int row = 1; row < Iterations; ++row) {
            replay_split<0, Config::kBodySlots, 16>(head, tail);
            TTI_INCRWC(0, 2, 0, 0);
        }
    }
}
}  // namespace ckernel::sfpu::bf16::sqrt
