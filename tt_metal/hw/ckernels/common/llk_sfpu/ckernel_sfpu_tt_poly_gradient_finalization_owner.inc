#if defined(TT_SELECTED_ZONE_GRADIENT)
        constexpr bool gradient_finalized_in_evaluator = true;
#elif defined(ARCH_WORMHOLE) && defined(TT_WH_CORE_SAME_ROW_GRAD_FINALIZE)
        constexpr bool gradient_finalized_in_evaluator = true;
#elif defined(DST_COEFF_STORE) && DST_COEFF_ELIGIBLE && !defined(DST_COEFF_PROBE) && \
    (defined(TT_INLINE_PROGRAM_DST_COEFF_SINGLE_ROW_OVERLAY) || defined(TT_SELECTED_CORE_TOTAL_SINGLE_ROW)) && \
    (defined(TT_INLINE_PROGRAM_DST_COEFF_SAME_ROW_GRAD_FINALIZE) || \
     defined(TT_SELECTED_CORE_TOTAL_SAME_ROW_GRAD_FINALIZE))
        constexpr bool gradient_finalized_in_evaluator = sfpi::kDstCoeffApply;
#else
        constexpr bool gradient_finalized_in_evaluator = false;
#endif
