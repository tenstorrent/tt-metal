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

    for (int d = 0; d < 32; d++) {
        vFloat x_orig = dst_reg[d];

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
        vFloat result = numer * ckernel::sfpu::sfpu_reciprocal<false>(denom);

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

        result = apply_output_postcompose(result);

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
inline void piecewise_rational_lut_dispatch(const std::array<float, LUT_SIZE>& lut) {
    piecewise_rational_lut_N<NUM_DEGREE, DEN_DEGREE, NUM_SEGMENTS, LUT_SIZE>(lut);
}
