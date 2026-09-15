// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace ckernel::sfpu {
inline void init_refine_loadmacros() {
    // Slot 0 is unused by the unclamped exp grid. SETEXP reads VC, so its
    // sequence must override VC (not VB) with the loaded register.
    TTI_SFPSETEXP(127, 0, 12, 1);
    TTI_SFPLOADI(0, 0xA, 0x0004);
    TTI_SFPLOADI(0, 0x8, 0x0000);
    TTI_SFPCONFIG(0, 5, 0);
    // Final load/multiply/store: polynomial is in L4/L5. The macro uses the
    // loaded linear grid value as VB and captures its DST address for store.
    TTI_SFPLOADI(0, 0xA, 0x8500);
    TTI_SFPLOADI(0, 0x8, 0x1300);
    TTI_SFPCONFIG(0, 6, 0);
    TTI_SFPLOADI(0, 0xA, 0x8600);
    TTI_SFPLOADI(0, 0x8, 0x1300);
    TTI_SFPCONFIG(0, 7, 0);
}

inline void restore_grid_macro_instructions() {
    TTI_SFPMAD(12, 0, 13, 13, 0);
    TTI_SFP_STOCH_RND(0, 0, 0, 0, 14, 7);
}

template <int iterations>
inline void calculate_sdpa_exp_refine_loadmacro() {
    static_assert(iterations % 2 == 0);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_6);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_7);
    constexpr float a = -8.01081703839006e27f;
    constexpr float b = 5.39213985142374e28f;
    constexpr float c = -1.05653430577460e29f;
    constexpr float d = 1.38859037541378e29f;
    TTI_SFPLOADI(6, 0xA, sdpa_lo16(a));
    TTI_SFPLOADI(6, 0x8, sdpa_hi16(a));
    TTI_SFPLOADI(0, 0xA, sdpa_lo16(b));
    TTI_SFPLOADI(0, 0x8, sdpa_hi16(b));
    TTI_SFPCONFIG(0, 12, 0);
    TTI_SFPLOADI(0, 0xA, sdpa_lo16(c));
    TTI_SFPLOADI(0, 0x8, sdpa_hi16(c));
    TTI_SFPCONFIG(0, 13, 0);
    TTI_SFPLOADI(0, 0xA, sdpa_lo16(d));
    TTI_SFPLOADI(0, 0x8, sdpa_hi16(d));
    TTI_SFPCONFIG(0, 14, 0);
    TTI_SFPMUL(4, 0, p_sfpu::LCONST_0, 13, 0);
    TTI_SFPMUL(5, 0, p_sfpu::LCONST_0, 14, 0);
    TTI_REPLAY(8, 10, 1, 1);
    TTI_SFPLOADMACRO(6, 0, ADDR_MOD_6, 0);
    TTI_SFPLOADMACRO(7, 0, ADDR_MOD_6, 2);
    TTI_SFPMAD(2, 6, 12, 4, 0);
    TTI_SFPMAD(3, 6, 12, 5, 0);
    TTI_SFPMAD(2, 4, 13, 4, 0);
    TTI_SFPMAD(3, 5, 13, 5, 0);
    TTI_SFPMAD(2, 4, 14, 4, 0);
    TTI_SFPMAD(3, 5, 14, 5, 0);
    TTI_SFPLOADMACRO(8, 0, ADDR_MOD_6, 0);
    TTI_SFPLOADMACRO(13, 0, ADDR_MOD_7, 2);
#pragma GCC unroll 8
    for (int i = 2; i < iterations; i += 2) {
        lltt::replay(8, 10);
    }
    // UnitDelayKind is elapsed SFPU instructions, as configured by exp init.
    // Retire the final two scheduled multiplies/stores before reconfiguration.
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}
}  // namespace ckernel::sfpu
