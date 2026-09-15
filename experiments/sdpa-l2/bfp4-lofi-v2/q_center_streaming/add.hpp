// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#ifdef TRISC_PACK
namespace ckernel::sfpu {
template <int BASE>
inline void q_center_add_four_rows() {
    TTI_SFPLOAD(p_sfpu::LREG0, 3, ADDR_MOD_7, BASE);
    TTI_SFPLOAD(p_sfpu::LREG1, 3, ADDR_MOD_7, BASE + 2);
    // Two score tiles followed by two correction tiles:2*64 raw DST rows.
    TTI_SFPLOAD(p_sfpu::LREG2, 3, ADDR_MOD_7, BASE + 128);
    TTI_SFPLOAD(p_sfpu::LREG3, 3, ADDR_MOD_7, BASE + 130);
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LREG1, 0);
    TTI_SFPNOP;
    TTI_SFPSTORE(p_sfpu::LREG0, 3, ADDR_MOD_7, BASE);
    TTI_SFPSTORE(p_sfpu::LREG1, 3, ADDR_MOD_7, BASE + 2);
}
inline void q_center_add_face() {
    // Other SFPU routines can leave automatic destination increments live.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    q_center_add_four_rows<0>();
    q_center_add_four_rows<4>();
    q_center_add_four_rows<8>();
    q_center_add_four_rows<12>();
}
}  // namespace ckernel::sfpu
#endif
