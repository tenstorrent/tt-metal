// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_sfpu_recip.h"
#include "cmath_common.h"

namespace ckernel::sfpu {

#ifndef DISABLE_SFPLOADMACRO
namespace softsign_detail {

// Two rows in flight: row a in LREG0-3, row b in LREG4-7 (x then t2, y, t, then y1 and the final reciprocal).
constexpr std::uint32_t xa = p_sfpu::LREG0, ya = p_sfpu::LREG1, ta = p_sfpu::LREG2, ra = p_sfpu::LREG3;
constexpr std::uint32_t xb = p_sfpu::LREG4, yb = p_sfpu::LREG5, tb = p_sfpu::LREG6, rb = p_sfpu::LREG7;

// Macro 0 loads v into x and schedules x = |x|, then x = 1 * x + 1; it steps the row counter.
template <std::uint32_t x>
inline void head() {
    TTI_SFPLOADMACRO((0 << 2) | (x & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_6, x >> 2);
}

// Macros 1 and 2 reload v from the row at offset off into t and schedule t = r * t, then store t to that row.
template <std::uint32_t macro, std::uint32_t t, std::uint32_t off>
inline void tail() {
    TTI_SFPLOADMACRO((macro << 2) | (t & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_7, (off & 0x3fe) | (t >> 2));
}

// The two Newton steps of sfpu_reciprocal_iter<2, true>, the reciprocal kept in r. x = |v| + 1 >= 1, so the
// guard of the unnormalized form only sees inf and NaN, where both forms give the same words.
inline void newton() {
    TTI_SFPARECIP(0, xa, ya, 0);
    TTI_SFPARECIP(0, xb, yb, 0);
    TTI_SFPMAD(xa, ya, p_sfpu::LREG12, ta, 2);
    TTI_SFPMAD(xb, yb, p_sfpu::LREG12, tb, 2);
    TTI_SFPMAD(ta, ya, p_sfpu::LCONST_0, ra, 3);
    TTI_SFPMAD(tb, yb, p_sfpu::LCONST_0, rb, 3);
    TTI_SFPMAD(xa, ra, p_sfpu::LREG12, xa, 1);
    TTI_SFPMAD(xb, rb, p_sfpu::LREG12, xb, 1);
    TTI_SFPMAD(ra, xa, p_sfpu::LCONST_0, ra, 2);
    TTI_SFPMAD(rb, xb, p_sfpu::LCONST_0, rb, 2);
}

}  // namespace softsign_detail

// One 32-row tile: each pair's tails run behind the next pair's heads, so every issue slot carries work.
inline void calculate_softsign_tile_macro() {
    using namespace softsign_detail;
    head<xa>();
    head<xb>();
    TTI_SFPNOP;
    TTI_SFPNOP;
    newton();
    load_replay_buf<Exec>(0, 14, [] {
        head<xa>();
        head<xb>();
        tail<1, ta, -8u>();
        tail<2, tb, -6u>();
        newton();
    });
#pragma GCC unroll 14
    for (int p = 2; p < 16; p++) {
        TTI_REPLAY(0, 14, 0, 0);
    }
    tail<1, ta, -4u>();
    tail<2, tb, -2u>();
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}
#endif

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_softsign() {
#ifndef DISABLE_SFPLOADMACRO
    if constexpr (!APPROXIMATION_MODE && ITERATIONS == 32) {
        calculate_softsign_tile_macro();
    } else
#endif
    {
        // SFPU microcode
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat v = sfpi::dst_reg[0];
            sfpi::vFloat tmp = sfpi::abs(v) + 1.0f;
            tmp = sfpu_reciprocal<APPROXIMATION_MODE>(tmp);
            sfpi::dst_reg[0] = v * tmp;
            sfpi::dst_reg++;
        }
    }
}

template <bool APPROXIMATION_MODE>
void init_softsign() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
#ifndef DISABLE_SFPLOADMACRO
    if constexpr (!APPROXIMATION_MODE) {
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
        // Templates 0 to 3: |x|, 1 * x + 1, ra * v, rb * v.
        TTI_SFPABS(0, 0, 12, 1);
        TTI_SFPADD(p_sfpu::LCONST_1, 0, p_sfpu::LCONST_1, 13, 0);
        TTI_SFPMUL(softsign_detail::ra, 0, p_sfpu::LCONST_0, 14, 0);
        TTI_SFPMUL(softsign_detail::rb, 0, p_sfpu::LCONST_0, 15, 0);
        // Macro 0: abs now, add one cycle later on the loaded register.
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, ((0x80 | (1 << 3) | 5) << 8) | (4 + 0));
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, 0);
        TTI_SFPCONFIG(0, 4 + 0, 0);
        // Macros 1 and 2: multiply one cycle after the load, store two cycles after the multiply.
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (0x80 | (1 << 3) | 6) << 8);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, ((3 << 3) | 3) << 8);
        TTI_SFPCONFIG(0, 4 + 1, 0);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (0x80 | (1 << 3) | 7) << 8);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, ((3 << 3) | 3) << 8);
        TTI_SFPCONFIG(0, 4 + 2, 0);
        // Misc: default store format, delays in cycles.
        TTI_SFPCONFIG(0, 8, 1);
    }
#endif
}

}  // namespace ckernel::sfpu
