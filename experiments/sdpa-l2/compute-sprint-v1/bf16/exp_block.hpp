// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Uses stock exp_init's replay and macro coefficients. Contiguous eight BF16
// tiles share a single setup/drain; same load/mad/round/shift/store per lane.
#ifdef TRISC_PACK
namespace ckernel::sfpu {
inline void calculate_sprint_exp_block() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_7);
#pragma GCC unroll 16
    for (int block = 0; block < 16; ++block) {
        lltt::replay(0, 32);
    }
    TTI_SFPNOP;
    TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LREG4, 5);
    TTI_SFPNOP;
    TTI_SFPSHFT2(p_sfpu::LREG3, p_sfpu::LREG14, p_sfpu::LREG4, 5);
    TTI_SFPNOP;
    TTI_SFPNOP;
}
}
#endif
