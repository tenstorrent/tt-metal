// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

    for (int d = 0; d < 32; d++) {
        vFloat x_raw = dst_reg[d];
        vFloat x_orig = prepare_raw_domain_input(x_raw);

        vFloat x = x_orig;

        // Clamping unnecessary: segment cascade v_if(x >= boundary) naturally selects
        // the edge segment for out-of-range inputs. Removing saves SFPU registers.
        vFloat& x_clamped = x;

        // Segment 0: evaluate P(x) and Q(x) (no reciprocal yet)
        vFloat numer, denom;
        eval_rational_numer_denom<NUM_DEGREE, DEN_DEGREE>(
            &lut[COEFF_OFFSET],
            &lut[COEFF_OFFSET + NUM_COEFFS],
            x,
            numer,
            denom
        );

        // Segments 1..N-1: v_if selects the correct numer/denom pair
        unroll_segment_rational_deferred<1, NUM_DEGREE, DEN_DEGREE, NUM_SEGMENTS, LUT_SIZE>(
            lut,
            x_clamped,
            x,
            numer,
            denom
        );

        // ONE reciprocal for all segments — saves ~10 instructions per eliminated recip
        vFloat result = numer * rational_reciprocal(denom);

        result = apply_output_postcompose(result, x_orig);
        finalize_raw_domain_actions(x_raw, result);

        // bf16 dst: RNE-round before the store. SFPSTORE narrows fp32->bf16 by
        // truncation (RTZ) in hardware; rounding here (sfpstochrnd RND_EVEN)
        // makes the already-bf16 value lossless under SFPSTORE and recovers the
        // half-ULP that RTZ would otherwise drop (the ML-pass output bias).
        result = convert<vFloat16b>(result, RoundMode::Nearest);
        dst_reg[d] = result;
    }
