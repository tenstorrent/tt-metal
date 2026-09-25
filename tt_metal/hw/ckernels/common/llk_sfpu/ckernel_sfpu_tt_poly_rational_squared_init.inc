#if WH_SQUARED_ABS_DENOMINATOR_TTI_CANDIDATE
    ckernel::addr_mod_t{
        .srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2},
    }.set(ckernel::ADDR_MOD_6);
#endif
#if SQUARED_ABS_DENOMINATOR_TTI_CANDIDATE
    if constexpr (sfpi::kSquaredAbsDenominatorTtiReplay) {
        // Prgm0/L12 remains the reciprocal helper's immutable 2.0.  No other
        // programmable constant is consumed: the certificate proves the body
        // needs neither a clamp bound nor a special poison.
        ckernel::addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},
        }
            .set(ckernel::ADDR_MOD_6);
        sfpi::squared_abs_denominator_tti_lm_init();
    }
#endif
