// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared rational whole-tile evaluation. Caller retains LUT context and tile
// index for the optional two-destination-half coefficient preload schedule.

#ifdef TRISC_MATH
        // All degree parameters are constexpr — use them directly as template args.
        // No dispatch table needed; works for ANY (num_degree, den_degree) combination.
#if defined(TT_WH_PERIODIC_Q192)
        _llk_math_eltwise_sfpu_start_(0);
#if defined(TT_WH_PERIODIC_WINDOW_POOL) && defined(TT_WH_PERIODIC_WINDOW37_ACTIVE)
#ifdef TT_WH_PERIODIC_FLOAT_EXPANSION_PARITY
        sfpi::tt_q192_expansion_prefill<true>();
#else
        sfpi::tt_q192_prefill_window_constants();
#endif
#endif
        for (int d = 0; d < 32; ++d) {
            sfpi::vFloat a = sfpi::tt_periodic_reduce_q192_tan(d);
            sfpi::vFloat square = a * a;
            sfpi::vFloat numerator = LUT_DATA[2 + NUM_DEGREE];
#pragma GCC unroll 8
            for (int i = int(NUM_DEGREE) - 2; i >= 1; i -= 2) {
                numerator = numerator * square + LUT_DATA[2 + i];
            }
            numerator = numerator * a;
            sfpi::vFloat denominator = LUT_DATA[3 + NUM_DEGREE + DEN_DEGREE];
#pragma GCC unroll 8
            for (int i = int(DEN_DEGREE) - 2; i >= 0; i -= 2) {
                denominator = denominator * square + LUT_DATA[3 + NUM_DEGREE + i];
            }
            sfpi::vFloat y = numerator * ckernel::sfpu::sfpu_reciprocal_iter<2>(denominator);
            sfpi::vUInt state = sfpi::tt_q192_state(d);
            v_if((((state >> 23) & 0xffu) > 125u) && ((state & 1u) != 0u)) {
                y = -ckernel::sfpu::sfpu_reciprocal_iter<2>(y);
            }
            v_endif;
            sfpi::finalize_raw_domain_actions(sfpi::dst_reg[d], y);
#if defined(TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL)
            sfpi::finalize_encoded_raw_domain_actions(
                sfpi::dst_reg[d].mode<sfpi::DataLayout::U16>(), y);
#endif
            sfpi::dst_reg[d] = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        }
        _llk_math_eltwise_sfpu_done_();
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_BIASED_LOG_RATIONAL_EXP_1 || \
    defined(EVAL_METHOD_BIASED_LOG_RATIONAL_EXP_1)
        (void)p_lut;
#if defined(TT_TARGET_COMPOSITE_NATIVE_EVEN_POLYVAL_EXTERIOR)
        sfpi::target_composite_preload_native_even_polyval_coefficients_tile();
        sfpi::target_composite_prepare_native_even_polyval_exterior_tile();
#endif
#if BIASED_LOG_RATIONAL_EXP_TTI_CANDIDATE
        sfpi::biased_log_rational_exp_replay_tile();
#elif defined(TT_BIASED_LOG_COMMON_SFPI)
        _llk_math_eltwise_sfpu_start_(0);
        sfpi::biased_log_rational_exp_common_tile();
        _llk_math_eltwise_sfpu_done_();
#else
#error "biased-log rational exp requires a declared target execution graph"
#endif
#if defined(TT_TARGET_COMPOSITE_NATIVE_EVEN_POLYVAL_EXTERIOR)
        sfpi::target_composite_apply_native_even_polyval_exterior_tile();
#elif defined(TT_TARGET_COMPOSITE_NATIVE_EVEN_CLASS_REP)
        sfpi::target_composite_apply_native_even_class_representative_tile();
#endif
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_ABS_DENOMINATOR_RATIONAL || defined(ABS_DENOMINATOR_RATIONAL)
        (void)p_lut;
#if WH_ABS_DENOMINATOR_TTI_CANDIDATE
        sfpi::wh_abs_denominator_replay_tile();
#else
        if constexpr (sfpi::kAbsDenominatorTtiReplay) {
#if ABS_DENOMINATOR_TTI_CANDIDATE
            sfpi::abs_denominator_tti_replay_tile();
#endif
        } else {
#if defined(ARCH_WORMHOLE)
            _llk_math_eltwise_sfpu_start_(0);
#endif
            sfpi::abs_denominator_rational_eval();
#if defined(ARCH_WORMHOLE)
            _llk_math_eltwise_sfpu_done_();
#endif
        }
#endif
#elif TT_ACT_EVAL_KIND == TT_ACT_EVAL_SQUARED_ABS_DENOMINATOR_RATIONAL || \
    defined(SQUARED_ABS_DENOMINATOR_RATIONAL)
        (void)p_lut;
#if WH_SQUARED_ABS_DENOMINATOR_TTI_CANDIDATE
        sfpi::wh_squared_abs_denominator_replay_tile();
#else
        if constexpr (sfpi::kSquaredAbsDenominatorTtiReplay) {
#if SQUARED_ABS_DENOMINATOR_TTI_CANDIDATE
            sfpi::squared_abs_denominator_tti_replay_tile();
#endif
        } else {
#if defined(ARCH_WORMHOLE)
            _llk_math_eltwise_sfpu_start_(0);
#endif
            sfpi::squared_abs_denominator_rational_eval();
#if defined(ARCH_WORMHOLE)
            _llk_math_eltwise_sfpu_done_();
#endif
        }
#endif
#else
#if defined(ARCH_WORMHOLE)
        _llk_math_eltwise_sfpu_start_(0);
#endif
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
#if defined(ARCH_WORMHOLE)
        _llk_math_eltwise_sfpu_done_();
#endif
#endif
#endif

#ifdef FUSE_GRAD_MUL
#ifdef TRISC_MATH
        // Epilogue fusion: every evaluator temporary is dead here — twin of
        // the piecewise_generic.cpp insertion point.
#if defined(ARCH_WORMHOLE)
        _llk_math_eltwise_sfpu_start_(0);
#endif
        sfpi::fuse_grad_mul();
#if defined(ARCH_WORMHOLE)
        _llk_math_eltwise_sfpu_done_();
#endif
#endif
#endif
