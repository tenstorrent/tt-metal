// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE>
static inline void eval_rational_interleaved_numer_denom(
    const float* num_coeffs, const float* den_coeffs, vFloat x, vFloat& out_numer, vFloat& out_denom) {
    constexpr uint32_t MIN_DEG = (NUM_DEGREE < DEN_DEGREE) ? NUM_DEGREE : DEN_DEGREE;

    vFloat numer = num_coeffs[NUM_DEGREE];
    vFloat denom = den_coeffs[DEN_DEGREE];

    if constexpr (NUM_DEGREE > DEN_DEGREE) {
#pragma GCC unroll 16
        for (int i = NUM_DEGREE - 1; i >= static_cast<int>(DEN_DEGREE); i--) {
            numer = numer * x + num_coeffs[i];
        }
    } else if constexpr (DEN_DEGREE > NUM_DEGREE) {
#pragma GCC unroll 16
        for (int i = DEN_DEGREE - 1; i >= static_cast<int>(NUM_DEGREE); i--) {
            denom = denom * x + den_coeffs[i];
        }
    }

#pragma GCC unroll 16
    for (int i = MIN_DEG - 1; i >= 0; i--) {
        numer = numer * x + num_coeffs[i];
        denom = denom * x + den_coeffs[i];
    }

    out_numer = numer;
    out_denom = denom;
}


static inline vFloat rational_reciprocal(vFloat denom) {
#if defined(RATIONAL_RECIPROCAL_ZERO_ITER)
    // Raw 8-bit SFPARECIP seed, zero Newton steps — TTNN's own erfc divide.
    return ckernel::sfpu::sfpu_reciprocal_iter<0>(denom);
#elif defined(RATIONAL_RECIPROCAL_ONE_ITER)
    return ckernel::sfpu::sfpu_reciprocal_iter<1>(denom);
#else
    return ckernel::sfpu::sfpu_reciprocal_iter<2>(denom);
#endif
}
