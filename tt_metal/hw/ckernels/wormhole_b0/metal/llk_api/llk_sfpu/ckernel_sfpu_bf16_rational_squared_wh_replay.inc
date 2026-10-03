// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#define WH_SQUARED_ABS_DENOMINATOR_TTI_CANDIDATE 1
inline void wh_squared_abs_denominator_replay_tile() {
    // The stock two-Newton square already equals every declared terminal. Without
    // FUSE_GRAD_SINGLE_ROUNDING the derivative is rounded to BF16 before the separate gradient pass.
    constexpr uint32_t kBodySlots = 27u
        + 2u
        + 3u
        ;
    TT_REPLAY_RECORD(kBodySlots);
    TTI_SFPLOAD(0, 0, TT_ADDRMOD_DST_HOLD, 0);
    TTI_SFPLOAD(4, 0, TT_ADDRMOD_DST_HOLD, 64);  // grad (dest tile + 1), in the load-consumer gap
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPADDI(0x3f80, 0, 0);
    TTI_SFPNOP;
    TTI_SFPNOT(0, 0, 3, 0);
    TTI_SFPMOV(0, 0, 5, 0);  // keep den = |x| + 1 for its exponent-FF class
    TTI_SFPSETMAN(0, ckernel::p_sfpu::LCONST_neg1, 0, 0);
    TTI_SFPMAD(12, 0, 13, 1, 0);
    TTI_SFPSETMAN(0, 3, 3, 1);
    TTI_SFPMAD(1, 0, 14, 1, 0);
    TTI_SFPMULI(0x3f00, 3, 0);
    TTI_SFPMAD(0, 1, ckernel::p_sfpu::LCONST_1, 2, 0);
    TTI_SFPNOP;
    TTI_SFPMAD(1, 2, 1, 1, 0);
    TTI_SFPNOP;
    TTI_SFPMAD(0, 1, ckernel::p_sfpu::LCONST_1, 2, 0);
    TTI_SFPNOP;
    TTI_SFPMAD(1, 2, 1, 1, 0);
    TTI_SFPNOP;
    TTI_SFPMUL(1, 3, ckernel::p_sfpu::LCONST_0, 1, 0);
    TTI_SFPNOP;
    TTI_SFPSETSGN(0, 1, 1, 1);
    // (reciprocal * grad) * reciprocal is formed in FP32 and rounded to BF16 once below; scaling
    // before squaring keeps a normal product out of the FTZ squared reciprocal.
    TTI_SFPMUL(1, 4, ckernel::p_sfpu::LCONST_0, 6, 0);
    TTI_SFPNOP;
    TTI_SFPMUL(6, 1, ckernel::p_sfpu::LCONST_0, 1, 0);
    TTI_SFPNOP;
    // den * 0 is +0 for finite x and a positive NaN for infinite or NaN x.
    TTI_SFPMAD(5, ckernel::p_sfpu::LCONST_0, 1, 1, 0);
    TTI_SFPNOP;
    TTI_SFP_STOCH_RND(SFPSTOCHRND_RND_EVEN, 0, 1, 1, 1,
        SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPNOP;
    TTI_SFPSTORE(1, 0, TT_ADDRMOD_DST_ADV, 0);
    TT_REPLAY_REST(kBodySlots, 31);
    TT_REPLAY_TILE_EPILOGUE();
}
