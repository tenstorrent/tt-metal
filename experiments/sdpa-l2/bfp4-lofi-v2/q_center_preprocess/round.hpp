// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#ifdef TRISC_MATH
namespace ckernel::sfpu {
// Same qualified raw-register mechanics as center_preprocess, but RNE7 and
// BF16 output. BF16 inputs occupy DST tiles j/j+4; subtraction stays FP32.
template <int BASE>
inline void q_center_four_rows() {
    TTI_SFPLOAD(p_sfpu::LREG0, 2, ADDR_MOD_7, BASE);
    TTI_SFPLOAD(p_sfpu::LREG1, 2, ADDR_MOD_7, BASE + 2);
    TTI_SFPLOAD(p_sfpu::LREG2, 2, ADDR_MOD_7, BASE + 256);
    TTI_SFPLOAD(p_sfpu::LREG3, 2, ADDR_MOD_7, BASE + 258);
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG0, 2);
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LREG1, 2);
    TTI_SFPNOP;
    // RNE7 = (raw + 0xffff + ((raw >> 17) & 1)) & 0xfffe0000.
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_USHORT, 0xffff);
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_SHORT, 0xffff);
    TTI_SFPSHFT(17, p_sfpu::LREG5, p_sfpu::LREG5, 1);
    TTI_SFPSHFT((-17) & 0xfff, p_sfpu::LREG0, p_sfpu::LREG2, 5);
    TTI_SFPSHFT((-17) & 0xfff, p_sfpu::LREG1, p_sfpu::LREG3, 5);
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_USHORT, 1);
    TTI_SFPAND(0, p_sfpu::LREG6, p_sfpu::LREG2, 0);
    TTI_SFPAND(0, p_sfpu::LREG6, p_sfpu::LREG3, 0);
    TTI_SFPIADD(0, p_sfpu::LREG4, p_sfpu::LREG0, 4);
    TTI_SFPIADD(0, p_sfpu::LREG4, p_sfpu::LREG1, 4);
    TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG0, 4);
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG1, 4);
    TTI_SFPAND(0, p_sfpu::LREG5, p_sfpu::LREG0, 0);
    TTI_SFPAND(0, p_sfpu::LREG5, p_sfpu::LREG1, 0);
    TTI_SFPSTORE(p_sfpu::LREG0, 2, ADDR_MOD_7, BASE);
    TTI_SFPSTORE(p_sfpu::LREG1, 2, ADDR_MOD_7, BASE + 2);
}
inline void q_center_face() {
    q_center_four_rows<0>();
    q_center_four_rows<4>();
    q_center_four_rows<8>();
    q_center_four_rows<12>();
}
}  // namespace ckernel::sfpu
#endif
