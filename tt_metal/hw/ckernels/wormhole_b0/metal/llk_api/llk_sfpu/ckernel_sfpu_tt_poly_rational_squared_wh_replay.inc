#if defined(ARCH_WORMHOLE) && defined(USE_BF16) && \
    defined(TT_WH_SQUARED_ABS_DENOMINATOR_REPLAY) && !defined(RATIONAL_TTI_DISABLE) && \
    !defined(RATIONAL_RECIPROCAL_ZERO_ITER) && !defined(RATIONAL_RECIPROCAL_ONE_ITER)
#define WH_SQUARED_ABS_DENOMINATOR_TTI_CANDIDATE 1
inline void wh_squared_abs_denominator_replay_tile() {
    // The stock two-Newton square already equals every declared terminal.
    // Keep the derivative BF16 conversion before the existing gradient pass.
    TT_REPLAY_RECORD(27);
    TTI_SFPLOAD(0, 0, TT_ADDRMOD_DST_HOLD, 0);
    TTI_SFPNOP;
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPADDI(0x3f80, 0, 0);
    TTI_SFPNOP;
    TTI_SFPNOT(0, 0, 3, 0);
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
    TTI_SFPMUL(1, 1, ckernel::p_sfpu::LCONST_0, 1, 0);
    TTI_SFPNOP;
    TTI_SFP_STOCH_RND(SFPSTOCHRND_RND_EVEN, 0, 1, 1, 1,
        SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPNOP;
    TTI_SFPSTORE(1, 0, TT_ADDRMOD_DST_ADV, 0);
    TT_REPLAY_REST(27, 31);
    TT_REPLAY_TILE_EPILOGUE();
}
#else
#define WH_SQUARED_ABS_DENOMINATOR_TTI_CANDIDATE 0
#endif
