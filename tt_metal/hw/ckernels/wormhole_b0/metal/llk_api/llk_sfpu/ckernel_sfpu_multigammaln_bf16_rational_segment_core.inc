// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

template <uint32_t SEG, uint32_t NUM_DEGREE, uint32_t DEN_DEGREE, uint32_t NUM_SEGMENTS, uint32_t LUT_SIZE>
__attribute__((always_inline)) static inline void unroll_segment_rational_deferred(
    const std::array<float, LUT_SIZE>& lut,
    vFloat x_clamped,
    vFloat x,
    vFloat& numer,
    vFloat& denom
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
            );
        }
        v_endif;
        unroll_segment_rational_deferred<SEG + 1, NUM_DEGREE, DEN_DEGREE, NUM_SEGMENTS, LUT_SIZE>(
            lut,
            x_clamped,
            x,
            numer,
            denom
        );
    }
}
