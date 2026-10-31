// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "lltt.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool IS_MAX_OP = true, int ITERATIONS = 8>
inline void calculate_binary_max_min(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    uint offset0 = (dst_index_in0 * 32) << 1;
    uint offset1 = (dst_index_in1 * 32) << 1;
    uint offset2 = (dst_index_out * 32) << 1;

#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        // Swap and store maximum in lreg1, minimum in lreg0
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, offset0);
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, offset1);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        TT_SFPSTORE(IS_MAX_OP ? p_sfpu::LREG1 : p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, offset2);
    }
#else
    if (dst_index_out == dst_index_in0 || dst_index_out == dst_index_in1) {
        const uint oa = dst_index_out == dst_index_in0 ? offset0 : offset1;
        const uint ob = dst_index_out == dst_index_in0 ? offset1 : offset0;
        if constexpr (ITERATIONS % 4 == 0) {
            load_replay_buf<Exec>(0, 8, [oa, ob] {
#pragma GCC unroll 4
                for (int i = 0; i < 4; ++i) {
                    const int a = (i & 1) | ((i & 2) << 1);
                    TT_SFPLOADMACRO(((2 + (i & 1)) << 2) | (a & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_7, oa | (a >> 2));
                    TT_SFPLOAD(p_sfpu::LREG2 + (i & 1), InstrModLoadStore::DEFAULT, ADDR_MOD_6, ob);
                }
            });
#pragma GCC unroll 8
            for (int r = 1; r < ITERATIONS / 4; ++r) {
                lltt::replay(0, 8);
            }
        } else {
#pragma GCC unroll 32
            for (int i = 0; i < ITERATIONS; ++i) {
                const int a = (i & 1) | ((i & 2) << 1);
                TT_SFPLOADMACRO(((2 + (i & 1)) << 2) | (a & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_7, oa | (a >> 2));
                TT_SFPLOAD(p_sfpu::LREG2 + (i & 1), InstrModLoadStore::DEFAULT, ADDR_MOD_6, ob);
            }
        }
        if constexpr (ITERATIONS <= 8) {
            TTI_SFPNOP;
            TTI_SFPNOP;
        }
        return;
    }

    constexpr int b = p_sfpu::LREG2;
    constexpr int c = p_sfpu::LREG3;

#pragma GCC unroll 32
    for (int i = 0; i < ITERATIONS; ++i) {
        int a = i & 1;  // alternate between p_sfpu::LREG0 and p_sfpu::LREG1
        TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_7, offset0 | (a >> 2));
        TT_SFPLOAD(b, InstrModLoadStore::DEFAULT, ADDR_MOD_7, offset1);
        TT_SFPLOADMACRO((1 << 2) | (c & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_6, offset2 | (c >> 2));
    }

    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
}

template <bool IS_MAX_OP = true, bool IS_UNSIGNED = false, int ITERATIONS = 8>
inline void calculate_binary_max_min_int32(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    uint offset0 = (dst_index_in0 * 32) << 1;
    uint offset1 = (dst_index_in1 * 32) << 1;
    uint offset2 = (dst_index_out * 32) << 1;

#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        // Swap and store maximum in lreg1, minimum in lreg0 (or reversed if unsigned)
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, offset0);
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, offset1);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, IS_UNSIGNED ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);

        // Conditionally swap again to fix the cases where SFPSWAP got the result backwards
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_SWAP);
        TTI_SFPENCC(0, 0, 0, 0);

        TT_SFPSTORE(IS_MAX_OP ? p_sfpu::LREG1 : p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_6, offset2);
    }
#else
    static_assert(ITERATIONS >= 2);
    constexpr int p0 = p_sfpu::LREG0;
    constexpr int p1 = p_sfpu::LREG1;
    constexpr int q0 = p_sfpu::LREG2;
    constexpr int q1 = p_sfpu::LREG3;
    constexpr int t0 = IS_MAX_OP ? q0 : p0;
    constexpr int t1 = IS_MAX_OP ? q1 : p1;
    constexpr uint cc = IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0;

    if (dst_index_out != dst_index_in0 && dst_index_out != dst_index_in1) {
#pragma GCC unroll 0
        for (int i = 0; i < ITERATIONS; ++i) {
            TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, offset0 + 2 * i);
            TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, offset2 + 2 * i);
        }
        offset0 = offset2;
    }
    const uint op = dst_index_out == dst_index_in1 ? offset1 : offset0;
    const uint oq = dst_index_out == dst_index_in1 ? offset0 : offset1;

    load_replay_buf<Exec>(0, 8, [op, oq] {
        TT_SFPLOADMACRO((2 << 2) | (p0 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, op | (p0 >> 2));
        TT_SFPLOADMACRO((0 << 2) | (q0 & 3), InstrModLoadStore::INT32, ADDR_MOD_6, oq | (q0 >> 2));
        TTI_SFPSETCC(0, t1, 0, cc);
        TTI_SFPENCC(0, 0, 0, 0);
        TT_SFPLOADMACRO((2 << 2) | (p1 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, op | (p1 >> 2));
        TT_SFPLOADMACRO((1 << 2) | (q1 & 3), InstrModLoadStore::INT32, ADDR_MOD_6, oq | (q1 >> 2));
        TTI_SFPSETCC(0, t0, 0, cc);
        TTI_SFPENCC(0, 0, 0, 0);
    });

#pragma GCC unroll 16
    for (int i = 1; i < ITERATIONS / 2; ++i) {
        lltt::replay(0, 8);
    }
    if constexpr (ITERATIONS & 1) {
        lltt::replay(0, 4);
    }

    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPSETCC(0, (ITERATIONS & 1) ? t0 : t1, 0, cc);
    TTI_SFPENCC(0, 0, 0, 0);
#endif
}

template <bool IS_MAX_OP = true>
inline void binary_max_min_init() {
#ifndef DISABLE_SFPLOADMACRO
    constexpr int b = p_sfpu::LREG2;

    // InstructionTemplate[0]
    TTI_SFPSWAP(0, b, 12, IS_MAX_OP ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // mod1=9 means set VD=max and VC=min

    // InstructionTemplate[1]
    TTI_SFPSHFT2(0, 0, 13, 6);  // SFPSHFT2_MOD1_SHFT_IMM

    TTI_SFPSWAP(0, p_sfpu::LREG3, 14, IS_MAX_OP ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);

    // Macro 0
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (1 << 3) | 4;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (3 << 3) | 5;
        constexpr uint store_bits = 0;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }

    // Macro 1
    {
        constexpr uint simple_bits = 0;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0;
        constexpr uint store_bits = 0x00 | 0x40 | (2 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }

    {
        constexpr uint simple_bits = 0x80 | 0x00 | (1 << 3) | 4;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0;
        constexpr uint store_bits = 0x00 | 0x00 | (3 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 2, 0);
    }

    {
        constexpr uint simple_bits = 0x80 | 0x00 | (1 << 3) | 6;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0;
        constexpr uint store_bits = 0x00 | 0x00 | (3 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 3, 0);
    }

    TTI_SFPCONFIG(0x3f0, 8, 1);
#endif
}

template <bool IS_MAX_OP = true, bool IS_UNSIGNED = false>
inline void binary_max_min_int32_init() {
#ifndef DISABLE_SFPLOADMACRO
    constexpr int p0 = p_sfpu::LREG0;
    constexpr int p1 = p_sfpu::LREG1;
    constexpr uint mode = IS_MAX_OP ^ IS_UNSIGNED ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX;

    TTI_SFPSWAP(0, p0, 12, mode);
    TTI_SFPSWAP(0, p1, 13, mode);

    TTI_SFPSHFT2(0, 0, 14, 6);

    {
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (4 << 3) | 6;
        constexpr uint store_bits = 0;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | (0x80 | 0x00 | (2 << 3) | 4));
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | (0x80 | 0x00 | (2 << 3) | 5));
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }

    {
        constexpr uint simple_bits = 0;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (6 << 3) | 6;
        constexpr uint store_bits = 0x00 | 0x40 | (7 << 3) | 3;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 2, 0);
    }

    TTI_SFPCONFIG(0x770, 8, 1);
#endif
}

}  // namespace sfpu
}  // namespace ckernel
