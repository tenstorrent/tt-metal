// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#ifdef SDPA_DIAG_EXP_MODE
#error "Sprint C refiner is only for the original cheaper-subtraction C recipe."
#endif
#if defined(TRISC_MATH) || defined(TRISC_PACK)
namespace ckernel::sfpu {
// Exact C biased cubic coefficients and original Horner MAD sequence.
// Only instruction scheduling changes: load macros combine SETEXP and final
// MUL/STORE; original grid value is reloaded from DST, never re-rounded.
template <int iterations>
inline void sprint_c_exp_refine_loadmacro() {
    static_assert(iterations % 2 == 0);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_6);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_7);
    constexpr float a = -8.106702647031e27f;
    constexpr float b = 5.446591617222e28f;
    constexpr float c = -1.069245718291e29f;
    constexpr float d = 1.399030410717e29f;
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

template <uint32_t scale_fp32>
inline void sprint_c_init_exp_grid() {
#ifndef SDPA_SPRINT_C_REFINE_HOIST
    init_sdpa_refine_loadmacros();
#endif
    restore_sdpa_grid_macro_instructions();
    init_sdpa_exp_grid<scale_fp32>();
}
}  // namespace ckernel::sfpu
#endif
#define init_sdpa_exp_grid sprint_c_init_exp_grid
#define calculate_sdpa_exp_stream_effective sprint_c_exp_refine_loadmacro
