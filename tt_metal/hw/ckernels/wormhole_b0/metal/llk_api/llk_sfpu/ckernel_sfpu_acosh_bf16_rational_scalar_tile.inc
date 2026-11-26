// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
    for (int d = 0; d < 32; d++) {
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

        // The segment cascade selects the edge segment for an out-of-range input.
        vFloat& x_clamped = x;

#if defined(RATIONAL_NUM_PARITY_ODD) && defined(RATIONAL_DEN_PARITY_EVEN)
        // Compute x² once, shared across all segments
        vFloat x2 = x * x;
#endif

        // Segment 0: evaluate P(x) and Q(x) (no reciprocal yet)
        vFloat numer, denom;
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

        // One reciprocal for all segments.
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

        // BF16 DEST: round to nearest here, as SFPSTORE truncates.
        result = convert<vFloat16b>(result, RoundMode::Nearest);
        dst_reg[d] = result;
    }
