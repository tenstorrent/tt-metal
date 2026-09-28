#if defined(TT_SELECTED_CORE_TTI_SQUARE_AFFINE_RECONSTRUCTION)
template <uint32_t J>
inline void poly_tti_square_affine_rung() {
    // Prgm0 is owned by the typed overlay's reciprocal, and that suffix may
    // use every allocatable LREG.  Make each recorded row independent of both
    // programmable-register and cross-row LREG state: c[D-2] was loaded into
    // L3 in the preamble, and every lower addend is streamed through that same
    // now-dead scratch immediately before its consuming MAD.
    constexpr uint32_t addend_reg = ckernel::p_sfpu::LREG3;
    if constexpr (J > 0u) {
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG3,
            SFPLOADI_MOD0_UPPER,
            (poly_tti_addend_bits(J) >> 16));
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG3,
            SFPLOADI_MOD0_LOWER,
            (poly_tti_addend_bits(J) & 0xffffu));
    } else {
        TTI_SFPNOP;
    }
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG1,
        addend_reg,
        ckernel::p_sfpu::LREG2,
        0);
}

inline void poly_tti_replay_tile_square_affine() {
    // Unlike the ordinary polynomial replay, this schedule cannot keep its
    // below-head addends in L7..L4 across rows: the same-row Mills/reciprocal
    // tail is allowed to consume every allocatable LREG.  Every addend is
    // consequently streamed into L3 inside the recorded body below.  Do not
    // run poly_tti_pin_coeffs() here -- those eight per-tile LOADI issues were
    // dead (the recorded body never reads the pins) and only delayed the
    // required typed suffix.  This is shape/resource driven and changes no
    // recorded arithmetic or raw-class action.
    TT_REPLAY_RECORD(TT_SELECTED_CORE_TTI_SQUARE_AFFINE_BODY_SLOTS);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_HOLD, 0);
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG3,
        SFPLOADI_MOD0_UPPER,
        (poly_tti_addend_bits(0) >> 16));
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG1,
        0);
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG3,
        SFPLOADI_MOD0_LOWER,
        (poly_tti_addend_bits(0) & 0xffffu));
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG13,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG14,
        ckernel::p_sfpu::LREG2,
        0);
    poly_tti_square_affine_rung<0>();
    poly_tti_square_affine_rung<1>();
    poly_tti_square_affine_rung<2>();
    poly_tti_square_affine_rung<3>();
    poly_tti_square_affine_rung<4>();
    poly_tti_square_affine_rung<5>();
    poly_tti_square_affine_rung<6>();
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG4,
        SFPLOADI_MOD0_UPPER,
        (TT_SELECTED_CORE_TTI_SQUARE_AFFINE_BIAS_BITS >> 16));
    TTI_SFPLOADI(
        ckernel::p_sfpu::LREG4,
        SFPLOADI_MOD0_LOWER,
        (TT_SELECTED_CORE_TTI_SQUARE_AFFINE_BIAS_BITS & 0xffffu));
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG2,
        0);
    auto same_row_suffix = []() {
        // The recorded core deliberately leaves its unrounded result in the
        // verifier-fixed L2 and does not advance or overwrite DST.  Reload the
        // original coordinate from this same row, bridge L2 into the existing
        // typed overlay, and perform the one final store only after every zone
        // and encoded-raw action has retired.
        {
            vFloat y = l_reg[LRegs::LReg2];
            vFloat x_raw = dst_reg[0];
            finalize_raw_domain_actions(x_raw, y);
            // End the numeric-tail live range at an architectural register.
            // The encoded classifier below reloads the still-untouched DST
            // row only after the Mills/exp/reciprocal graph has retired.
            l_reg[LRegs::LReg2] = y;
        }
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL)
        {
            vUInt raw_u16 = dst_reg[0].mode<DataLayout::U16>();
            vFloat y = l_reg[LRegs::LReg2];
            finalize_encoded_raw_domain_actions(raw_u16, y);
            l_reg[LRegs::LReg2] = y;
        }
#endif
        vFloat y = l_reg[LRegs::LReg2];
#ifdef USE_BF16
        y = convert<vFloat16b>(y, RoundMode::Nearest);
#endif
        dst_reg[0] = y;
        dst_reg++;
    };
    same_row_suffix();
    TT_REPLAY_REST_WITH_SAME_ROW_SUFFIX(
        TT_SELECTED_CORE_TTI_SQUARE_AFFINE_BODY_SLOTS, 31, same_row_suffix());
    TT_REPLAY_TILE_EPILOGUE();
}
#endif
