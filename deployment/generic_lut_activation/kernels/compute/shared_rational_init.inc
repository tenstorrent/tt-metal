// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared initialization: include once after init_sfpu and before the tile loop.

#ifdef TRISC_MATH
    // Initialize reciprocal SFPU function (needed for rational evaluation)
    // Platform-independent init for reciprocal
    ckernel::sfpu::sfpu_reciprocal_init<false>();
#if BIASED_LOG_RATIONAL_EXP_TTI_CANDIDATE
    // The closed form does not use reciprocal helper's 2.0 after init. The
    // programmable lanes own immutable d2/d1/e2 while L3..L6 own immutable
    // n3/n2/n1/n0. Every replay iteration therefore enters with identical
    // coefficient state.
    sfpi::vConstFloatPrgm0 = sfpi::TT_BIASED_LOG_DEN[2];
    sfpi::vConstFloatPrgm1 = sfpi::TT_BIASED_LOG_DEN[1];
    sfpi::vConstFloatPrgm2 = sfpi::TT_BIASED_LOG_NUM[4];
    // L3..L6 are loop-carried constants.  The selected replay certificate
    // proves it never writes them; the optional class-merge certificate uses
    // only dead L0/L1/L2/L7 scratch.  No SFPU instruction between tiles
    // touches these lanes, so pin them once for the complete kernel.
    sfpi::biased_log_rational_exp_preload_kernel();
    ckernel::addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ckernel::ADDR_MOD_6);
    sfpi::biased_log_rational_exp_lm_init();
#endif
#if ABS_DENOMINATOR_TTI_CANDIDATE
    if constexpr (sfpi::kAbsDenominatorTtiReplay) {
        // Prgm0/L12 remains the reciprocal helper's immutable 2.0.  The
        // certificate-derived terminal bound occupies otherwise-unused
        // Prgm1/L13, whose read-only lane makes vec-min/max non-destructive.
        sfpi::vConstFloatPrgm1 = __builtin_bit_cast(
            float,
            (uint32_t)TT_TARGET_BH_BF16_ABS_DENOMINATOR_BOUND_BITS);
        ckernel::addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},
        }
            .set(ckernel::ADDR_MOD_6);
        sfpi::abs_denominator_tti_lm_init();
    }
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
#if RATIONAL_TTI_CANDIDATE
    if constexpr (sfpi::kRationalTtiReplay) {
        // TTI replay fast path: SFPSTORE walks dst via ADDR_MOD_6 dest += 2
        // (single element per body). Programmed once; nothing else in this
        // kernel touches ADDR_MOD_6.
        ckernel::addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},
        }
            .set(ckernel::ADDR_MOD_6);
        if constexpr (sfpi::kRatTtiLmTail) {
            // Arm the tail-fold macro (templates + sequence reg 0 + Misc).
            // After sfpu_reciprocal_init (backdoor capture cannot clobber the
            // prgm constants) and >= 1 instruction before the first
            // SFPLOADMACRO (audit L9 WAR rule — the tile loop is far away).
            sfpi::rational_tti_lm_init();
        }
    }
#endif
#endif
