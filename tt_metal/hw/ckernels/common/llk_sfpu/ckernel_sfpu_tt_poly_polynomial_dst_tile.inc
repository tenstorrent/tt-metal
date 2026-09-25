// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Main body: mirrors piecewise_generic_lut_specialized_N_dual for the gated
// shape (no basis, no range reduction) with parked coefficient fetches.
template <uint32_t POLY_DEG_T, uint32_t NUM_SEG_T, uint32_t LUT_SIZE_T>
inline void piecewise_generic_lut_dst_coeff(const std::array<float, LUT_SIZE_T>& lut) {
    (void)lut;
    dstcoeff::park_all();
#if defined(TT_INLINE_PROGRAM_DST_COEFF_SINGLE_ROW_OVERLAY) || defined(TT_SELECTED_CORE_TOTAL_SINGLE_ROW)
#if defined(TT_INLINE_PROGRAM_DST_COEFF_SINGLE_ROW_OVERLAY)
    static_assert(
        TT_PROGRAM_DST_COEFF_DUAL_PEAK_LIVE > TT_PROGRAM_SFPU_LREG_CAPACITY,
        "single-row DST schedule requires a verifier-derived dual live-set overflow");
#if defined(TT_INLINE_PROGRAM_DST_COEFF_SAME_ROW_GRAD_FINALIZE)
    static_assert(
        TT_PROGRAM_DST_COEFF_DUAL_PEAK_LIVE == 9u && TT_PROGRAM_AUX_LEAF_MAX_LIVE == 3u &&
            TT_PROGRAM_DST_COEFF_LATE_FINALIZER_MAX_LIVE == 3u,
        "same-row gradient finalization requires the exact verifier-derived peak-9 shape");
    static_assert(
        TT_PROGRAM_DST_COEFF_GRAD_FINALIZE_PEAK_LIVE == 2u &&
            TT_PROGRAM_DST_COEFF_GRAD_FINALIZE_PEAK_LIVE <= TT_PROGRAM_SFPU_LREG_CAPACITY,
        "same-row gradient finalizer exceeds the verified SFPU live-set capacity");
#if !defined(FUSE_GRAD_MUL)
#error "same-row gradient finalization requires the fused-gradient ABI"
#endif
#endif
#endif
#if defined(TT_SELECTED_CORE_TOTAL_SAME_ROW_GRAD_FINALIZE) && !defined(FUSE_GRAD_MUL)
#error "typed same-row gradient finalization requires the fused-gradient ABI"
#endif
#if defined(ARCH_WORMHOLE) && defined(TT_WH_DST_COEFF_ROW_UNROLL)
#pragma GCC unroll 2
#endif
    for (int d = 0; d < 32; d++) {
        vFloat x_raw = dst_reg[d];
#if defined(TT_DOMAIN_ACTION_LOWER_CLAMP_RAW_NEG_EXP_FF_INGRESS)
        vFloat x = prepare_raw_domain_input(x_raw, dst_reg[d].mode<DataLayout::U16>());
#else
        vFloat x = prepare_raw_domain_input(x_raw);
#endif
        vFloat r;
#if defined(TT_WH_DST_CERTIFIED_MIRROR) || (!defined(ARCH_WORMHOLE) && \
    (defined(TT_INLINE_PROGRAM_DST_COEFF_EVEN_MIRROR_FOLD) || defined(TT_SELECTED_CORE_TOTAL_DST_EVEN_MIRROR_FOLD)))
        vFloat x_folded = setsgn(x, 0);
        dstcoeff::Transport::eval_seg_single<dstcoeff::kEvenMirrorFirstSegment>(x_folded, r);
        dstcoeff::Transport::cascade_single<dstcoeff::kEvenMirrorFirstSegment + 1>(x_folded, r);
#else
        dstcoeff::Transport::eval_seg_single<0>(x, r);
        dstcoeff::Transport::cascade_single<1>(x, r);
#endif
#if defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC)
        dstcoeff::asym_apply_dst(x, r);
#endif
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL) || defined(TT_INLINE_PROGRAM_RAW_ZERO_BOUNDARY) || \
    defined(TT_INLINE_PROGRAM_RESULT_TERMINAL_RAW_OVERRIDE)
        // Complete the ordinary overlay/reconstruction before materializing
        // encoded classifier state.  The three-argument overload preserves
        // this exact order for other schedules, while the already-selected
        // single-row schedule makes the lifetime boundary explicit so SFPI
        // cannot retain raw_u16 across the inlined reconstruction body.
        finalize_raw_domain_actions(x_raw, r);
        vUInt raw_u16 = dst_reg[d].mode<DataLayout::U16>();
        finalize_encoded_raw_domain_actions(x_raw, raw_u16, r);
#else
        finalize_raw_domain_actions(x_raw, r);
#endif
#ifdef USE_BF16
        r = convert<vFloat16b>(r, RoundMode::Nearest);
#endif
#if defined(TT_INLINE_PROGRAM_DST_COEFF_SAME_ROW_GRAD_FINALIZE) || \
    defined(TT_SELECTED_CORE_TOTAL_SAME_ROW_GRAD_FINALIZE) || \
    (defined(ARCH_WORMHOLE) && defined(TT_WH_CORE_SAME_ROW_GRAD_FINALIZE))
        // Preserve the ABI's existing double rounding exactly: the selected
        // derivative is first rounded to BF16, then multiplied by the BF16
        // incoming gradient parked in DST tile 1, then rounded again.  Loading
        // grad only here keeps it out of Horner/reconstruction liveness.
        vFloat grad = dst_reg[32 + d];
        r = r * grad;
#ifdef USE_BF16
        r = convert<vFloat16b>(r, RoundMode::Nearest);
#endif
#endif
#if defined(TT_SELECTED_ZONE_GRADIENT)
        r = finalize_zone_gradient(dst_reg[d], dst_reg[d].mode<DataLayout::U16>(), r, dst_reg[32 + d]);
        r = convert<vFloat16b>(r, RoundMode::Nearest);
#endif
        dst_reg[d] = r;
    }
#else
    for (int d = 0; d < 32; d += 2) {
        vFloat x_raw1 = dst_reg[d];
        vFloat x_raw2 = dst_reg[d + 1];
#if defined(TT_DOMAIN_ACTION_LOWER_CLAMP_RAW_NEG_EXP_FF_INGRESS)
        vFloat x1 = prepare_raw_domain_input(x_raw1, dst_reg[d].mode<DataLayout::U16>());
        vFloat x2 = prepare_raw_domain_input(x_raw2, dst_reg[d + 1].mode<DataLayout::U16>());
#else
        vFloat x1 = prepare_raw_domain_input(x_raw1);
        vFloat x2 = prepare_raw_domain_input(x_raw2);
#endif
        vFloat r1, r2;
#if defined(TT_WH_DST_CERTIFIED_MIRROR) || (!defined(ARCH_WORMHOLE) && \
    (defined(TT_INLINE_PROGRAM_DST_COEFF_EVEN_MIRROR_FOLD) || defined(TT_SELECTED_CORE_TOTAL_DST_EVEN_MIRROR_FOLD)))
        vFloat x_folded1 = setsgn(x1, 0);
        vFloat x_folded2 = setsgn(x2, 0);
        dstcoeff::Transport::eval_seg_dual<dstcoeff::kEvenMirrorFirstSegment>(x_folded1, x_folded2, r1, r2);
        dstcoeff::Transport::cascade_dual<dstcoeff::kEvenMirrorFirstSegment + 1>(x_folded1, x_folded2, r1, r2);
#else
        dstcoeff::Transport::eval_seg_dual<0>(x1, x2, r1, r2);
        dstcoeff::Transport::cascade_dual<1>(x1, x2, r1, r2);
#endif
#if defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC)
        dstcoeff::asym_apply_dst(x1, r1);
        dstcoeff::asym_apply_dst(x2, r2);
#endif
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL) || defined(TT_INLINE_PROGRAM_RAW_ZERO_BOUNDARY) || \
    defined(TT_INLINE_PROGRAM_RESULT_TERMINAL_RAW_OVERRIDE)
        // Reload each preserved raw lane only after the dual selected Horner
        // and reconstruction temporaries have retired.  This retains the
        // existing coefficient residency and two-row evaluator schedule.
        vUInt raw_u16_1 = dst_reg[d].mode<DataLayout::U16>();
        vUInt raw_u16_2 = dst_reg[d + 1].mode<DataLayout::U16>();
        finalize_raw_domain_actions(x_raw1, raw_u16_1, r1);
        finalize_raw_domain_actions(x_raw2, raw_u16_2, r2);
#else
        finalize_raw_domain_actions(x_raw1, r1);
        finalize_raw_domain_actions(x_raw2, r2);
#endif
#ifdef USE_BF16
        r1 = convert<vFloat16b>(r1, RoundMode::Nearest);
        r2 = convert<vFloat16b>(r2, RoundMode::Nearest);
#endif
#if defined(TT_SELECTED_ZONE_GRADIENT)
        r1 = finalize_zone_gradient(dst_reg[d], dst_reg[d].mode<DataLayout::U16>(), r1, dst_reg[32 + d]);
        r1 = convert<vFloat16b>(r1, RoundMode::Nearest);
        r2 = finalize_zone_gradient(dst_reg[d + 1], dst_reg[d + 1].mode<DataLayout::U16>(), r2, dst_reg[33 + d]);
        r2 = convert<vFloat16b>(r2, RoundMode::Nearest);
#endif
        dst_reg[d] = r1;
        dst_reg[d + 1] = r2;
    }
#endif
}
