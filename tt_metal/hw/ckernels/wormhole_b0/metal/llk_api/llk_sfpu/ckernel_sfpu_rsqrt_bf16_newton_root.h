// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_rsqrt_bf16_replay_window.h"
// WH cbrt runs stock's kernel, defined in ckernel_sfpu_cbrt.h, which includes this file.
namespace ckernel::sfpu {
template <bool, bool, int>
inline void calculate_cube_root();
}
namespace ckernel::sfpu::bf16::rsqrt {
constexpr uint32_t kNewtonRsqrtCompactSlots = 28u;

inline void newton_rsqrt_compact_row_head() {
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, 0);                                  // x
    TTI_SFPEXEXP(0, p_sfpu::LREG1, p_sfpu::LREG6, sfpi::SFPEXEXP_MOD1_NODEBIAS);   // biased exponent
    TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);                                // WH shifts its destination in place
    TTI_SFPSHFT(0xFFF, 0, p_sfpu::LREG0, 1);                                       // bits(x) >> 1
    TTI_SFPIADD(0, p_sfpu::LREG12, p_sfpu::LREG0, 6);                              // y = MAGIC - i
    TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);  // xy
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG5, 1);                                // -y: WH SFPMUL has no negate
    TTI_SFPMUL(p_sfpu::LREG5, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);  // c = -y*xy
    TTI_SFPLOADI(p_sfpu::LREG3, 0, 0x7F80);                                        // +Inf
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG14, p_sfpu::LREG2, p_sfpu::LREG5, 0);  // C2 + c
    TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG4, 2);                                 // x bits
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG5, p_sfpu::LREG13, p_sfpu::LREG2, 0);     // t = c*(C2+c) + C1
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG4, 6);                                // inf_bits - x_bits
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);   // y *= t
    TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG5, 1);                                 // |x|
    TTI_SFPSETCC(0, p_sfpu::LREG6, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);                // exponent zero
}

inline void newton_rsqrt_compact_row_tail() {
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG7, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);  // stock's y * 1.5
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);  // x < 0
    TTI_SFPLOADI(p_sfpu::LREG0, 0, 0x7FC0);
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_SFPSETCC(0, p_sfpu::LREG4, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);  // x != +Inf
    TTI_SFPSETCC(0, p_sfpu::LREG5, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);  // x != +-0
    TTI_SFPCOMPC(0, 0, 0, 0);
    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);  // +-Inf at +-0, 0 at +Inf
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
    static_assert(Config::kKind == 1);
    static_assert(Iterations == 8 || Iterations == 32);
    static_assert(Config::kBodySlots == kNewtonRsqrtCompactSlots);
    {
        TTI_SFPLOADI(p_sfpu::LREG7, 0, 0x3FC0);  // 1.5
        auto head = [&] { newton_rsqrt_compact_row_head(); };
        auto tail = [&] { newton_rsqrt_compact_row_tail(); };
        record_split<0, Config::kBodySlots, 16>(head, tail);
        TTI_INCRWC(0, 2, 0, 0);
#pragma GCC unroll 8
        for (int row = 1; row < Iterations; ++row) {
            replay_split<0, Config::kBodySlots, 16>(head, tail);
            TTI_INCRWC(0, 2, 0, 0);
        }
    }
}
}  // namespace ckernel::sfpu::bf16::rsqrt
