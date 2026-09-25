#if defined(TT_WH_SYMMETRIC_EXP_POOL) && defined(TRISC_MATH)
#if !defined(ARCH_WORMHOLE) || !defined(USE_BF16) || defined(TT_SELECTED_CORE_TOTAL_NEEDS_RECIPROCAL)
#error "symmetric exp coefficient pool requires its WH BF16 non-reciprocal owner"
#endif
    sfpi::vConstFloatPrgm0 = TT_SELECTED_CORE_EXP_COEFFS[3];
    sfpi::vConstFloatPrgm1 = TT_SELECTED_CORE_EXP_COEFFS[2];
    sfpi::vConstFloatPrgm2 = TT_SELECTED_CORE_EXP_COEFFS[1];
#endif
