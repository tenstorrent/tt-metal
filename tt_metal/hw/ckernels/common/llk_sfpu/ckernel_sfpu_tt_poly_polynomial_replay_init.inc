        // Poly-cascade replay preamble: coefficients into only the
        // structurally allocated programmable const registers once (the poly
        // path otherwise never touches them), and the ADDR_MOD_6 dst walk for
        // the body's final
        // SFPSTORE: +4 rows (2 vectors) for the 2-element plain/parity
        // interleaves, +2 rows for the single-element signed-abs body.
        // kPolyTtiReplay proves this write set is disjoint from every row
        // owned by the typed overlay's reciprocal primitive. In particular,
        // no write below can clobber sfpu_reciprocal_init's last-writer state.
#if POLY_TTI_SHAPE_LOG_SQUARE
        sfpi::vConstFloatPrgm0 = 0.69314718055994530942f * 0x1p-23f;
        sfpi::vConstFloatPrgm1 = __builtin_bit_cast(
            float, sfpi::TT_SELECTED_LOG_SQUARE_LOG_COEFF_BITS[2]);
        sfpi::vConstFloatPrgm2 = __builtin_bit_cast(
            float, sfpi::TT_SELECTED_LOG_SQUARE_LOG_COEFF_BITS[1]);
#elif POLY_TTI_SHAPE_PARITY_EVEN
        // Parity-even stride-2 top coefficients: c[TOP], c[TOP-2], c[TOP-4].
        sfpi::vConstFloatPrgm1 = sfpi::poly_tti_c(sfpi::kPolyTtiParityTop);
        sfpi::vConstFloatPrgm2 = sfpi::poly_tti_c(sfpi::kPolyTtiParityTop - 2u);
        sfpi::vConstFloatPrgm0 = sfpi::poly_tti_c(sfpi::kPolyTtiParityTop - 4u);
#else
        sfpi::vConstFloatPrgm1 = sfpi::poly_tti_c(POLY_DEGREE);
        sfpi::vConstFloatPrgm2 = sfpi::poly_tti_c(POLY_DEGREE - 1);
        if constexpr (sfpi::kPolyTtiPrgmWriteMask & sfpi::kPolyTtiPrgm0Mask) {
            sfpi::vConstFloatPrgm0 = sfpi::poly_tti_c(POLY_DEGREE - 2);
        }
#endif
        ckernel::addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = (sfpi::kPolyTtiPlain2x || sfpi::kPolyTtiParityEven2x) ? 4 : 2},
        }
            .set(ckernel::ADDR_MOD_6);
