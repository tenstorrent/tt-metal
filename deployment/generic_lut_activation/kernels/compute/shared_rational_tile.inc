// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared rational whole-tile evaluation. Caller retains LUT context and tile
// index for the optional two-destination-half coefficient preload schedule.

#ifdef TRISC_MATH
        // All degree parameters are constexpr — use them directly as template args.
        // No dispatch table needed; works for ANY (num_degree, den_degree) combination.
#if TT_ACT_EVAL_KIND == TT_ACT_EVAL_BIASED_LOG_RATIONAL_EXP_1 || \
    defined(EVAL_METHOD_BIASED_LOG_RATIONAL_EXP_1)
        (void)p_lut;
#if defined(TT_TARGET_COMPOSITE_NATIVE_EVEN_POLYVAL_EXTERIOR)
        sfpi::target_composite_preload_native_even_polyval_coefficients_tile();
        sfpi::target_composite_prepare_native_even_polyval_exterior_tile();
#endif
#if BIASED_LOG_RATIONAL_EXP_TTI_CANDIDATE
        sfpi::biased_log_rational_exp_replay_tile();
#else
#error "biased-log rational exp requires its certified BH BF16 replay"
#endif
#if defined(TT_TARGET_COMPOSITE_NATIVE_EVEN_POLYVAL_EXTERIOR)
        sfpi::target_composite_apply_native_even_polyval_exterior_tile();
#elif defined(TT_TARGET_COMPOSITE_NATIVE_EVEN_CLASS_REP)
        sfpi::target_composite_apply_native_even_class_representative_tile();
#endif
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_ABS_DENOMINATOR_RATIONAL || defined(ABS_DENOMINATOR_RATIONAL)
        (void)p_lut;
        if constexpr (sfpi::kAbsDenominatorTtiReplay) {
#if ABS_DENOMINATOR_TTI_CANDIDATE
            sfpi::abs_denominator_tti_replay_tile();
#endif
        } else {
            sfpi::abs_denominator_rational_eval();
        }
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_SQUARED_ABS_DENOMINATOR_RATIONAL || \
    defined(SQUARED_ABS_DENOMINATOR_RATIONAL)
        (void)p_lut;
        if constexpr (sfpi::kSquaredAbsDenominatorTtiReplay) {
#if SQUARED_ABS_DENOMINATOR_TTI_CANDIDATE
            sfpi::squared_abs_denominator_tti_replay_tile();
#endif
        } else {
            sfpi::squared_abs_denominator_rational_eval();
        }
#else
        sfpi::piecewise_rational_lut_dispatch<num_degree, den_degree, num_segments, lut_size>(
            *p_lut
#if TT_RATIONAL_DST_COEFF
            // Default: re-store the coefficient rows every tile (same
            // acquire window as the reads). DST_COEFF_PRELOAD_ONCE writes
            // only the first two tiles (one per SyncHalf dest half) — see
            // the ZEROACC note in the block comment above.
#ifdef DST_COEFF_PRELOAD_ONCE
            ,
            tile < 2
#else
            ,
            true
#endif
#endif
        );
#endif
#endif

#ifdef FUSE_GRAD_MUL
#ifdef TRISC_MATH
        // Epilogue fusion: every evaluator temporary is dead here — twin of
        // the piecewise_generic.cpp insertion point.
        sfpi::fuse_grad_mul();
#endif
#endif
