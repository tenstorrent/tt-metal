// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Copied from the qualified bfp4_round prototype; only residual liveness and
// component DST addressing differ. No existing source is included or modified.
#ifdef TRISC_MATH
namespace ckernel::sfpu {

// The fixed-register body deliberately does not mix live SFPI compiler-managed
// vectors with raw instructions. LREG0..7 are scratch; no constant is rewritten.
// ADDR_MOD_7 is the zero-increment SFPU address modifier initialized by startup.
inline void bfp4_max_r2_r3() {
    // In VEC_MIN_MAX mode VC gets max and VD gets min: retain max in LREG2.
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, 1);
    TTI_SFPNOP;
}

template <int ROTATIONS>
inline void bfp4_rotate_r2_to_r3() {
    TTI_SFPSHFT2(0, p_sfpu::LREG2, p_sfpu::LREG3, 3);
    TTI_SFPNOP;
    for (int i = 1; i < ROTATIONS; ++i) {
        TTI_SFPSHFT2(0, p_sfpu::LREG3, p_sfpu::LREG3, 3);
        TTI_SFPNOP;
    }
}

template <bool FP32_DST, int BASE, int COMPONENT, bool LAST>
inline void residual_round_stage() {
    // Each load is four rows x eight columns of one parity. Their pair covers
    // four independent native BFP groups of 16 adjacent columns, not 32 columns.
    constexpr int dst_format = FP32_DST ? 3 : 2;  // explicit FP32 / BF16
    TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG2, 1);
    TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG3, 1);
    bfp4_max_r2_r3();

    // A cyclic max butterfly broadcasts each group's max to all eight lanes.
    // SHFLROR1 never crosses an eight-lane subgroup (one face row).
    bfp4_rotate_r2_to_r3<1>();
    bfp4_max_r2_r3();
    bfp4_rotate_r2_to_r3<2>();
    bfp4_max_r2_r3();
    bfp4_rotate_r2_to_r3<4>();
    bfp4_max_r2_r3();

    // Let E=floor(log2(max(abs(x)))). The BFP4 grid is delta=2^(E-2).
    // magic=2^(E+21) has FP32 ulp=delta. Two SEPARATE rounded additions
    // therefore implement RNE to this grid. LREG4 holds magic.
    TTI_SFPEXEXP(0, p_sfpu::LREG2, p_sfpu::LREG4, 1);  // biased exponent
    TTI_SFPIADD(21, p_sfpu::LREG4, p_sfpu::LREG4, 5);  // immediate, no CC
    TTI_SFPSETEXP(0, p_sfpu::LCONST_1, p_sfpu::LREG4, 0);

    // cap=7*delta=1.75*2^E. Use the maximum's exponent, avoiding a multiply.
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_FLOATB, 0x3fe0);
    TTI_SFPSETEXP(0, p_sfpu::LREG3, p_sfpu::LREG2, 2);
    TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG5, 1);
    TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG6, 1);
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, p_sfpu::LREG5, 0);
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LREG4, p_sfpu::LREG6, 0);
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, p_sfpu::LREG5, 2);
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LREG4, p_sfpu::LREG6, 2);

    // Copy cap before destructive min/max. MOV hides the MAD->SWAP hazard;
    // an explicit NOP follows each SWAP as required by the Blackhole ISA.
    TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG7, 0);
    TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG5, 1);
    TTI_SFPNOP;
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG6, 1);
    TTI_SFPNOP;
    // Preserve signed current residual in LREG0/1. LREG2/3 are dead scratch
    // after saturation, so use them for signed component values.
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
    TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG3, 0);
    TTI_SFPSETSGN(0, p_sfpu::LREG5, p_sfpu::LREG2, 0);
    TTI_SFPSETSGN(0, p_sfpu::LREG6, p_sfpu::LREG3, 0);
    // One tile occupies 64 DST rows. Current face offset comes from the
    // enclosing SFPU wrapper; component offsets select separate DST tiles.
    TTI_SFPSTORE(p_sfpu::LREG2, dst_format, ADDR_MOD_7, BASE + COMPONENT * 64);
    TTI_SFPSTORE(p_sfpu::LREG3, dst_format, ADDR_MOD_7, BASE + COMPONENT * 64 + 2);
    if constexpr (!LAST) {
        TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG0, 2);
        TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LREG1, 2);
        TTI_SFPNOP;
    }
}

template <bool FP32_DST, int BASE>
inline void residual_rne5_tail() {
    // RNE to five significant bits, applied to the live FP32 residual's bits:
    // (raw + 0x3ffff + ((raw >> 19) & 1)) & 0xfff80000.
    // This handles signed ordinary values and zero without float arithmetic.
    // Native BFP8 packing subsequently rounds to its shared-exponent grid.
    constexpr int dst_format = FP32_DST ? 3 : 2;
    TTI_SFPLOADI(p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_SHORT, 0xffff);
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_SHORT, 0xffff);
    TTI_SFPSHFT((-14) & 0xfff, p_sfpu::LREG4, p_sfpu::LREG4, 1);
    TTI_SFPSHFT(19, p_sfpu::LREG5, p_sfpu::LREG5, 1);
    TTI_SFPSHFT((-19) & 0xfff, p_sfpu::LREG0, p_sfpu::LREG2, 5);
    TTI_SFPSHFT((-19) & 0xfff, p_sfpu::LREG1, p_sfpu::LREG3, 5);
    TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_USHORT, 1);
    TTI_SFPAND(0, p_sfpu::LREG6, p_sfpu::LREG2, 0);
    TTI_SFPAND(0, p_sfpu::LREG6, p_sfpu::LREG3, 0);
    TTI_SFPIADD(0, p_sfpu::LREG4, p_sfpu::LREG0, 4);
    TTI_SFPIADD(0, p_sfpu::LREG4, p_sfpu::LREG1, 4);
    TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG0, 4);
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG1, 4);
    TTI_SFPAND(0, p_sfpu::LREG5, p_sfpu::LREG0, 0);
    TTI_SFPAND(0, p_sfpu::LREG5, p_sfpu::LREG1, 0);
    TTI_SFPSTORE(p_sfpu::LREG0, dst_format, ADDR_MOD_7, BASE + 64);
    TTI_SFPSTORE(p_sfpu::LREG1, dst_format, ADDR_MOD_7, BASE + 66);
}

template <bool FP32_DST, int COMPONENTS, bool SECOND_B8, int BASE>
inline void residual_round_four_face_rows() {
    constexpr int dst_format = FP32_DST ? 3 : 2;
    TTI_SFPLOAD(p_sfpu::LREG0, dst_format, ADDR_MOD_7, BASE);
    TTI_SFPLOAD(p_sfpu::LREG1, dst_format, ADDR_MOD_7, BASE + 2);
    residual_round_stage<FP32_DST, BASE, 0, false>();
    if constexpr (SECOND_B8) {
        residual_rne5_tail<FP32_DST, BASE>();
    } else {
        residual_round_stage<FP32_DST, BASE, 1, COMPONENTS == 2>();
        if constexpr (COMPONENTS == 3) {
            residual_round_stage<FP32_DST, BASE, 2, true>();
        }
    }
}

template <bool FP32_DST, int COMPONENTS, bool SECOND_B8>
inline void residual_round_face() {
    static_assert(COMPONENTS == 2 || COMPONENTS == 3);
    static_assert(!SECOND_B8 || COMPONENTS == 2);
    residual_round_four_face_rows<FP32_DST, COMPONENTS, SECOND_B8, 0>();
    residual_round_four_face_rows<FP32_DST, COMPONENTS, SECOND_B8, 4>();
    residual_round_four_face_rows<FP32_DST, COMPONENTS, SECOND_B8, 8>();
    residual_round_four_face_rows<FP32_DST, COMPONENTS, SECOND_B8, 12>();
}
}  // namespace ckernel::sfpu
#endif
