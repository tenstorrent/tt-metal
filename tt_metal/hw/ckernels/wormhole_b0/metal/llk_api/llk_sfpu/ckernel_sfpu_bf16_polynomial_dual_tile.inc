// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

template <uint32_t SEG, uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
__attribute__((always_inline)) inline void unroll_segment_dual(
    const std::array<float, LUT_SIZE>& lut,
    vFloat x1c,
    vFloat x2c,
    vFloat x1,
    vFloat x2,
    vFloat& r1,
    vFloat& r2
) {
    if constexpr (SEG < NUM_SEGMENTS) {
        constexpr uint32_t CPS = POLY_DEGREE + 1;
        constexpr uint32_t CO = NUM_SEGMENTS + 1;
        constexpr uint32_t DEG = SEGMENT_DEGREES[SEG];
        {
            vFloat tmp1, tmp2;
            eval_polynomial_dual<DEG>(&lut[CO + SEG * CPS], x1, x2, tmp1, tmp2);
            vFloat b = lut[SEG];
            v_if(x1c >= b) { r1 = tmp1; }
            v_endif;
            v_if(x2c >= b) { r2 = tmp2; }
            v_endif;
        }
        unroll_segment_dual<SEG + 1, POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(
            lut,
            x1c,
            x2c,
            x1,
            x2,
            r1,
            r2
        );
    }
}

template <uint32_t POLY_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
inline void piecewise_generic_lut_specialized_N_dual(const std::array<float, LUT_SIZE>& lut) {
    [[maybe_unused]] constexpr uint32_t COEFFS_PER_SEGMENT = POLY_DEGREE + 1;
    constexpr uint32_t COEFF_OFFSET = NUM_SEGMENTS + 1;

    for (int d = 0; d < 32; d += 2) {
        vFloat x_raw1 = dst_reg[d];
        vFloat x_raw2 = dst_reg[d + 1];
        vFloat x_orig1 = prepare_raw_domain_input(x_raw1);
        vFloat x_orig2 = prepare_raw_domain_input(x_raw2);

        vFloat x1 = x_orig1;
        vFloat x2 = x_orig2;

        // Dual-eval: skip clamping to avoid SFPU register spill from vec_min_max temporaries.
        // Segment boundaries still gate correctly — out-of-range x just picks the edge segment.
        vFloat& x1_clamped = x1;
        vFloat& x2_clamped = x2;

        // Segment 0: dual evaluation for ILP
        vFloat result1, result2;
        eval_polynomial_dual<SEGMENT_DEGREES[0]>(&lut[COEFF_OFFSET], x1, x2, result1, result2);

        // Segments 1..NUM_SEGMENTS-1 via recursive unrolling
        unroll_segment_dual<1, POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(
            lut,
            x1_clamped,
            x2_clamped,
            x1,
            x2,
            result1,
            result2
        );

        finalize_raw_domain_actions(x_raw1, result1);
        finalize_raw_domain_actions(x_raw2, result2);
        finalize_encoded_raw_domain_actions(dst_reg[d].mode<DataLayout::U16>(), result1);
        finalize_encoded_raw_domain_actions(dst_reg[d + 1].mode<DataLayout::U16>(), result2);

        // bf16 dst: RNE-round before the store (see single-eval note above).
        // grad * f'(x) is formed at full SFPU precision and rounded to BF16 once.
        result1 = result1 * dst_reg[32 + d];
        result2 = result2 * dst_reg[33 + d];
        result1 = convert<vFloat16b>(result1, RoundMode::Nearest);
        result2 = convert<vFloat16b>(result2, RoundMode::Nearest);
        dst_reg[d] = result1;
        dst_reg[d + 1] = result2;
    }
}
