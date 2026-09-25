// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
    for (int d = 0; d < 32; d++) {
#if defined(TT_MIRRORED_RATIONAL_TYPED_RAW_NEG_NAN_TERMINAL) || \
    (defined(TT_INLINE_PROGRAM_RATIONAL_LATE_RAW_SCHEDULE) && !defined(ARCH_WORMHOLE))
        vUInt raw_u16 = dst_reg[d].mode<DataLayout::U16>();
#endif
        vFloat x_raw = dst_reg[d];
        vFloat x_orig = prepare_raw_domain_input(x_raw);

#if defined(RANGE_REDUCTION_EXP)
        constexpr float EXP_OVERFLOW = 88.5f;
        constexpr float EXP_UNDERFLOW = -88.5f;
        vFloat x;
        vInt k_int;
        exp_reduce(x_orig, x, k_int);
#elif defined(RANGE_REDUCTION_TRIG)
        vFloat x;
        vInt q_int;
        trig_reduce(x_orig, x, q_int);
#elif defined(RANGE_REDUCTION_LOG)
        vFloat x;
        vInt e_int;
        log_reduce(x_orig, x, e_int);
#else
        vFloat x = x_orig;
#endif
#if defined(PRECOMPOSE_INPUT_AFFINE)
        x = PRECOMPOSE_INPUT_A * x + PRECOMPOSE_INPUT_B;
#endif
#if defined(TT_RATIONAL_COORDINATE_BOUND)
        x = rational_bound_coordinate(x);
#endif

        // Clamping unnecessary: segment cascade v_if(x >= boundary) naturally selects
        // the edge segment for out-of-range inputs. Removing saves SFPU registers.
        vFloat& x_clamped = x;

#if defined(RATIONAL_NUM_PARITY_ODD) && defined(RATIONAL_DEN_PARITY_EVEN)
        // Compute x² once, shared across all segments
        vFloat x2 = x * x;
#endif

        // Segment 0: evaluate P(x) and Q(x) (no reciprocal yet)
        vFloat numer, denom;
#if defined(TT_WH_PARITY_NUMERATOR_PIN)
        eval_rational_parity_numer_denom<NUM_DEGREE, DEN_DEGREE, true>(
            &lut[COEFF_OFFSET], &lut[COEFF_OFFSET + NUM_COEFFS], x, x2, numer, denom, pin_top);
#else
        eval_rational_numer_denom<NUM_DEGREE, DEN_DEGREE>(
            &lut[COEFF_OFFSET],
            &lut[COEFF_OFFSET + NUM_COEFFS],
            x,
            numer,
            denom
#if defined(RATIONAL_NUM_PARITY_ODD) && defined(RATIONAL_DEN_PARITY_EVEN)
            ,
            x2
#endif
        );

#endif
        // Segments 1..N-1: v_if selects the correct numer/denom pair
        unroll_segment_rational_deferred<1, NUM_DEGREE, DEN_DEGREE, NUM_SEGMENTS, LUT_SIZE>(
            lut,
            x_clamped,
            x,
            numer,
            denom
#if defined(RATIONAL_NUM_PARITY_ODD) && defined(RATIONAL_DEN_PARITY_EVEN)
            ,
            x2
#endif
        );

        // ONE reciprocal for all segments — saves ~10 instructions per eliminated recip
        vFloat result = numer * rational_reciprocal(denom);

#if defined(RANGE_REDUCTION_EXP)
        v_if(x_orig > EXP_OVERFLOW) { result = std::numeric_limits<float>::infinity(); }
        v_elseif(x_orig < EXP_UNDERFLOW) { result = 0.0f; }
        v_else { result = exp_expand(result, k_int); }
        v_endif;
#elif defined(RANGE_REDUCTION_TRIG)
        result = trig_expand(result, q_int);
#elif defined(RANGE_REDUCTION_LOG)
        v_if(x_orig < 0.0f) { result = std::numeric_limits<float>::quiet_NaN(); }
        v_elseif(x_orig == 0.0f) { result = -std::numeric_limits<float>::infinity(); }
        v_else { result = log_expand(result, e_int); }
        v_endif;
#endif

        result = apply_output_postcompose(result, x_orig);
        finalize_raw_domain_actions(x_raw, result);

#if defined(TT_MIRRORED_RATIONAL_TYPED_RAW_NEG_NAN_TERMINAL)
        negative_nan_class_terminal<TT_MIRRORED_RATIONAL_RAW_NEG_NAN_RESULT, false>(raw_u16, result);
#endif

#if defined(TT_INLINE_PROGRAM_RATIONAL_LATE_RAW_SCHEDULE)
        static_assert(
            TT_PROGRAM_RATIONAL_LATE_RAW_CORE_PEAK_LIVE <= TT_PROGRAM_RATIONAL_LATE_RAW_LREG_CAPACITY &&
                TT_PROGRAM_RATIONAL_LATE_RAW_SUFFIX_PEAK_LIVE <= TT_PROGRAM_RATIONAL_LATE_RAW_LREG_CAPACITY,
            "rational selected core or raw-only suffix exceeds SFPU capacity");
        // The ordinary selected Horner/reconstruction has retired, while the
        // BF16 destination source row is still byte-exact.  Materialize raw
        // classifier state only now and let it take terminal precedence over
        // the decoded +Inf policy before the unchanged RNE/store.
#if defined(ARCH_WORMHOLE)
        vUInt raw_u16 = dst_reg[d].mode<DataLayout::U16>();
#endif
        finalize_encoded_raw_domain_actions(raw_u16, result);
#endif

        // bf16 dst: RNE-round before the store. SFPSTORE narrows fp32->bf16 by
        // truncation (RTZ) in hardware; rounding here (sfpstochrnd RND_EVEN)
        // makes the already-bf16 value lossless under SFPSTORE and recovers the
        // half-ULP that RTZ would otherwise drop (the ML-pass output bias).
#ifdef USE_BF16
        result = convert<vFloat16b>(result, RoundMode::Nearest);
#endif
#if defined(TT_BF16_RAW_UINT16_OUTPUT)
        dst_reg[d].mode<DataLayout::U16>() = mathematical_aggregate_bf16_word(d, result);
#else
        dst_reg[d] = result;
#endif
    }
