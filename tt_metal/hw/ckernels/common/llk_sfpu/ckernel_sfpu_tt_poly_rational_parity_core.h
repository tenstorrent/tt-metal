// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Include inside namespace sfpi. Exact canonical interleaved parity Horner.
template <uint32_t NUM_DEGREE, uint32_t DEN_DEGREE, bool PIN_TOP = false>
inline void eval_rational_parity_numer_denom(
    const float* num_coeffs,
    const float* den_coeffs,
    vFloat x,
    vFloat x2,
    vFloat& out_numer,
    vFloat& out_denom,
    vFloat pin_top = vFloat(0.0f)) {
    constexpr int NUM_TOP = (NUM_DEGREE % 2 == 1) ? NUM_DEGREE : NUM_DEGREE - 1;
    constexpr int DEN_TOP = (DEN_DEGREE % 2 == 0) ? DEN_DEGREE : DEN_DEGREE - 1;
    constexpr int NUM_STEPS = (NUM_TOP - 1) / 2;
    constexpr int DEN_STEPS = DEN_TOP / 2;

    vFloat numer;
    if constexpr (PIN_TOP) {
        numer = pin_top;
    } else {
        numer = num_coeffs[NUM_TOP];
    }
    vFloat denom = den_coeffs[DEN_TOP];

    if constexpr (NUM_STEPS > DEN_STEPS) {
#pragma GCC unroll 16
        for (int k = 0; k < NUM_STEPS - DEN_STEPS; k++) {
            numer = numer * x2 + num_coeffs[NUM_TOP - 2 * (k + 1)];
        }
    } else if constexpr (DEN_STEPS > NUM_STEPS) {
#pragma GCC unroll 16
        for (int k = 0; k < DEN_STEPS - NUM_STEPS; k++) {
            denom = denom * x2 + den_coeffs[DEN_TOP - 2 * (k + 1)];
        }
    }

    constexpr int MIN_STEPS = (NUM_STEPS < DEN_STEPS) ? NUM_STEPS : DEN_STEPS;
    constexpr int NUM_POS = NUM_TOP - 2 * ((NUM_STEPS > DEN_STEPS) ? (NUM_STEPS - DEN_STEPS) : 0);
    constexpr int DEN_POS = DEN_TOP - 2 * ((DEN_STEPS > NUM_STEPS) ? (DEN_STEPS - NUM_STEPS) : 0);

#pragma GCC unroll 16
    for (int k = 1; k <= MIN_STEPS; k++) {
        numer = numer * x2 + num_coeffs[NUM_POS - 2 * k];
        denom = denom * x2 + den_coeffs[DEN_POS - 2 * k];
    }

    out_numer = numer * x;  // odd parity: P(x) = x * Horner_result
    out_denom = denom;
}

// Selected scalar path, preserving the canonical pin lifetime and raw-load
// placement. Upstream caller owns SFPU start/done and the 32-row DST window.
template <uint32_t NumDegree, uint32_t DenDegree, uint32_t BoundBits, int RawClass, bool PinTop, bool RawEarly>
inline void mirrored_parity_rational_tile(const float* num_coeffs, const float* den_coeffs) {
    static_assert(NumDegree == 11 && DenDegree == 4);
    static_assert(RawEarly != PinTop);
    vFloat pin_top = 0.0f;
    if constexpr (PinTop) {
        l_reg[LRegs::LReg7] = vFloat(num_coeffs[NumDegree]);
        pin_top = l_reg[LRegs::LReg7];
    }
#if defined(ARCH_WORMHOLE)
#pragma GCC unroll 32
#endif
    for (int d = 0; d < 32; d++) {
        vUInt raw_u16;
        if constexpr (RawEarly) {
            raw_u16 = dst_reg[d].template mode<DataLayout::U16>();
        }
        vFloat x_raw = dst_reg[d];
        vFloat x2 = x_raw * x_raw;
        vFloat numer, denom;
        eval_rational_parity_numer_denom<NumDegree, DenDegree, PinTop>(
            num_coeffs, den_coeffs, x_raw, x2, numer, denom, pin_top);
        vFloat result = numer * ckernel::sfpu::sfpu_reciprocal_iter<2>(denom);
        mirrored_class_terminals<BoundBits, true>(x_raw, result);
        if constexpr (!RawEarly) {
            raw_u16 = dst_reg[d].template mode<DataLayout::U16>();
        }
        negative_nan_class_terminal<RawClass, !RawEarly>(raw_u16, result);
        result = convert<vFloat16b>(result, RoundMode::Nearest);
        dst_reg[d] = result;
    }
    if constexpr (PinTop) {
        l_reg[LRegs::LReg7] = pin_top;
    }
}
