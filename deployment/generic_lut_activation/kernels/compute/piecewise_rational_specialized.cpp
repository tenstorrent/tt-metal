// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

/**
 * Template-recursive unrolling for piecewise rational evaluation
 * with deferred reciprocal optimization.
 *
 * Instead of computing P(x)/Q(x) per segment (3 reciprocals for 3 segments,
 * all executing on all 32 SIMD lanes due to predicated v_if execution),
 * we evaluate P(x) and Q(x) inside v_if to select the correct pair,
 * then do ONE reciprocal outside all v_ifs.
 *
 * Saves ~10 SFPU instructions per eliminated reciprocal (sfparecip +
 * 4 Newton-Raphson sfpmad + control flow).
 *
 * CPS (coeffs per segment) = (NUM_DEGREE + 1) + (DEN_DEGREE + 1) because
 * LUT stores numerator then denominator coefficients for each segment.
 */

#pragma once

// ============================================================================
// Recursive unroller — deferred reciprocal (evaluates P and Q, no division)
// ============================================================================

template <uint32_t SEG, uint32_t NUM_DEGREE, uint32_t DEN_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
__attribute__((always_inline)) inline void unroll_segment_rational_deferred(
    const std::array<float, LUT_SIZE>& lut,
    vFloat x_clamped,
    vFloat x,
    vFloat& numer,
    vFloat& denom
#if defined(RATIONAL_NUM_PARITY_ODD) && defined(RATIONAL_DEN_PARITY_EVEN)
    ,
    vFloat x2
#endif
) {
    if constexpr (SEG < NUM_SEGMENTS) {
        constexpr uint32_t NUM_COEFFS = NUM_DEGREE + 1;
        constexpr uint32_t CPS = NUM_COEFFS + DEN_DEGREE + 1;
        constexpr uint32_t CO = NUM_SEGMENTS + 1;
        v_if(x_clamped >= lut[SEG]) {
            eval_rational_numer_denom<NUM_DEGREE, DEN_DEGREE>(
                &lut[CO + SEG * CPS],
                &lut[CO + SEG * CPS + NUM_COEFFS],
                x,
                numer,
                denom
#if defined(RATIONAL_NUM_PARITY_ODD) && defined(RATIONAL_DEN_PARITY_EVEN)
                ,
                x2
#endif
            );
        }
        v_endif;
        unroll_segment_rational_deferred<SEG + 1, NUM_DEGREE, DEN_DEGREE, NUM_SEGMENTS, LUT_SIZE>(
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
    }
}

// ============================================================================
// Main function with deferred reciprocal
// ============================================================================

template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_rational_lut_N(const std::array<float, LUT_SIZE>& lut) {
    constexpr uint32_t NUM_COEFFS = NUM_DEGREE + 1;
    constexpr uint32_t CPS = NUM_COEFFS + DEN_DEGREE + 1;
    constexpr uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;

#if TT_RATIONAL_LREG_PIN_ELIGIBLE
    // BH + bf16 single-segment R7/4-class fast path: pin the two top numerator
    // coefficients in LReg4/5 for the whole 32-element dst loop (see the
    // block comment above eval_rational_interleaved_numer_denom_pinned in
    // piecewise_rational.cpp). Saves 2 in-body SFPLOADI pairs per element at a
    // bit-identical FMA sequence. RATIONAL_PIN_DISABLE compiles this out.
    if constexpr (NUM_SEGMENTS == 1 && rational_pin_shape_ok<NUM_DEGREE, DEN_DEGREE>) {
        // Load once per tile: each pin is one pre-loop SFPLOADI pair.
        l_reg[LRegs::LReg4] = vFloat(lut[COEFF_OFFSET + NUM_DEGREE]);
        l_reg[LRegs::LReg5] = vFloat(lut[COEFF_OFFSET + NUM_DEGREE - 1]);
        vFloat pin_n_top = l_reg[LRegs::LReg4];
        vFloat pin_n_next = l_reg[LRegs::LReg5];
        // Indexed dst_reg[d], never the dst_reg++ walk (wave-3 lesson: the raw
        // 32-iteration walk drifts the DST write counter at block boundaries).
        for (int d = 0; d < 32; d++) {
            vFloat x_raw = dst_reg[d];
            vFloat x_orig = prepare_raw_domain_input(x_raw);
            vFloat x = x_orig;
#if defined(TT_RATIONAL_COORDINATE_BOUND)
            x = rational_bound_coordinate(x);
#endif
            vFloat numer, denom;
            eval_rational_interleaved_numer_denom_pinned<NUM_DEGREE, DEN_DEGREE>(
                &lut[COEFF_OFFSET], &lut[COEFF_OFFSET + NUM_COEFFS], x, pin_n_top, pin_n_next, numer, denom);
            vFloat result = numer * rational_reciprocal_pinned(denom);
            // Gate excludes range reduction and pre/postcompose; USE_BF16 is a
            // gate precondition, so the RNE narrowing store is unconditional.
            finalize_raw_domain_actions(x_raw, result);
            result = convert<vFloat16b>(result, RoundMode::Nearest);
            dst_reg[d] = result;
        }
        // Write-back: pins stay live-out of the loop so the register
        // allocator can neither clobber nor rematerialize them (the
        // ckernel_sfpu_gelu.h l_reg idiom).
        l_reg[LRegs::LReg4] = pin_n_top;
        l_reg[LRegs::LReg5] = pin_n_next;
        return;
    }
#endif

    for (int d = 0; d < 32; d++) {
#if defined(TT_MIRRORED_RATIONAL_TYPED_RAW_NEG_NAN_TERMINAL) || defined(TT_INLINE_PROGRAM_RATIONAL_LATE_RAW_SCHEDULE)
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
        // Physical BF16 DST layout is sign[15], mantissa[14:8],
        // exponent[7:0]. This selects exactly the 127 negative NaNs.
        vUInt exponent_and_sign = raw_u16 & vUInt(0x80ffu);
        vUInt mantissa = raw_u16 & vUInt(0x7f00u);
        v_if(exponent_and_sign == vUInt(0x80ffu) && mantissa != 0u) {
            static_assert(
                TT_MIRRORED_RATIONAL_RAW_NEG_NAN_RESULT <= 4u, "mirrored rational raw terminal requires a typed class");
            if constexpr (TT_MIRRORED_RATIONAL_RAW_NEG_NAN_RESULT == 0u) {
                result = std::numeric_limits<float>::quiet_NaN();
            } else if constexpr (TT_MIRRORED_RATIONAL_RAW_NEG_NAN_RESULT == 1u) {
                result = std::numeric_limits<float>::infinity();
            } else if constexpr (TT_MIRRORED_RATIONAL_RAW_NEG_NAN_RESULT == 2u) {
                result = -std::numeric_limits<float>::infinity();
            } else if constexpr (TT_MIRRORED_RATIONAL_RAW_NEG_NAN_RESULT == 3u) {
                result = vFloat(0.0f);
            } else {
                result = setsgn(vFloat(0.0f), 1);
            }
        }
        v_endif;
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
        finalize_encoded_raw_domain_actions(raw_u16, result);
#endif

        // bf16 dst: RNE-round before the store. SFPSTORE narrows fp32->bf16 by
        // truncation (RTZ) in hardware; rounding here (sfpstochrnd RND_EVEN)
        // makes the already-bf16 value lossless under SFPSTORE and recovers the
        // half-ULP that RTZ would otherwise drop (the ML-pass output bias).
#ifdef USE_BF16
        result = convert<vFloat16b>(result, RoundMode::Nearest);
#endif
        dst_reg[d] = result;
    }
}

// ============================================================================
// Dispatcher
// ============================================================================

template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_rational_lut_dispatch(
    const std::array<float, LUT_SIZE>& lut
#if TT_RATIONAL_DST_COEFF
    ,
    bool dst_preload
#endif
) {
#if TT_RATIONAL_DST_COEFF
    // DST-resident coefficient store (see the block comment in
    // piecewise_rational.cpp). Requires fp32 dest accumulation — when the
    // host didn't flip the dest mode (DST_ACCUM_MODE false) the mechanism is
    // silently off and the proven sfpi path below runs unchanged.
    if constexpr (DST_ACCUM_MODE) {
        (void)lut;
        piecewise_rational_dst_coeff_tile(dst_preload);
        return;
    } else {
        (void)dst_preload;
    }
#endif
#if RATIONAL_TTI_CANDIDATE
    // TTI replay lowering for the single-segment parity rational (see the
    // kRationalTtiReplay block in piecewise_rational.cpp). Reads the constexpr
    // embedded LUT_DATA directly; when the gate is false the sfpi loop below
    // compiles unchanged.
    if constexpr (kRationalTtiReplay) {
        (void)lut;
#if defined(TT_RATIONAL_COORDINATE_BOUND)
        rational_bound_coordinate_tile();
#endif
        rational_tti_replay_tile();
        return;
    }
#endif
    piecewise_rational_lut_N<NUM_DEGREE, DEN_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut);
}
