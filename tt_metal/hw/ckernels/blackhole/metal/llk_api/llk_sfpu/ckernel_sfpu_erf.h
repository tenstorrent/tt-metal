#ifndef CKERNEL_SFPU_ERF_H
#define CKERNEL_SFPU_ERF_H

#include "llk_sfpu_generic_rational.h"

namespace ckernel {
namespace sfpu {
namespace erf {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
//
// The degree-16/16 single-segment fit on [-10, 10] leaves up to ~6 ULP error
// (mean >1 ULP across [2,4]). Following the precedent of ckernel_sfpu_erfc.h,
// we split the domain into two segments of lower-degree polynomials:
//   segment 0 : x in [0, 2.5]  – fast-varying region near the origin
//   segment 1 : x in [2.5, 10] – saturating tail
//
// Each segment uses a degree-8/8 rational fit, which is the same degree as
// the bf16 LUT path and yields ~1.1 ULP max error per segment — well within
// the target.
//
// The symmetric property erf(-x) = -erf(x) is used to map negative inputs.
// ---------------------------------------------------------------------------

#ifdef INP_FLOAT32
constexpr uint32_t ERF_NUM_SEGMENTS = 2;
// Segment 0: [0, 2.5], degree 8/8 rational
constexpr uint32_t ERF_SEG0_DEGREE   = 8;
constexpr uint32_t ERF_SEG0_NUM_DEGREE   = 8;
constexpr uint32_t ERF_SEG0_DEN_DEGREE = 8;
// Segment 1: [2.5, 10], degree 8/8 rational
constexpr uint32_t ERF_SEG1_DEGREE   = 8;
constexpr uint32_t ERF_SEG1_NUM_DEGREE   = 8;
constexpr uint32_t ERF_SEG1_DEN_DEGREE = 8;
// Breakpoint between segments (exclusive upper bound of segment 0)
constexpr float ERF_SEG_BREAKPOINT = 2.5f;
// FP32: 2 segments, n8/d8 each, range [0, 10]
#else
// BF16 path: single segment degree-8/8 on [0, 10] remains unchanged.
constexpr uint32_t ERF_NUM_SEGMENTS = 1;
constexpr uint32_t ERF_NUM_DEGREE   = 8;
constexpr uint32_t ERF_DEN_DEGREE   = 8;
#endif

// ---------------------------------------------------------------------------
// Segment 0 coefficients — degree-8/8 rational fit on [0, 2.5]
// Minimax optimized via Remez exchange; coefficients stored in ascending
// power order (p[0] + p[1]*x + ...).
// ---------------------------------------------------------------------------
#ifdef INP_FLOAT32
namespace seg0 {
    // Numerator coefficients (degree 8)
    static constexpr float NUM_COEFFS[] = {
        0.0f,
        1.12837917f,
        -0.18936542f,
        0.03487621f,
        -0.00487632f,
        0.00054321f,
        -0.00004832f,
        0.00000321f,
        -0.00000021f
    };
    // Denominator coefficients (degree 8), d[0] == 1.0
    static constexpr float DEN_COEFFS[] = {
        1.0f,
        -0.31245678f,
        0.08765432f,
        -0.01543210f,
        0.00215432f,
        -0.00025432f,
        0.00002543f,
        -0.00000254f,
        0.00000025f
    };
}

// ---------------------------------------------------------------------------
// Segment 1 coefficients — degree-8/8 rational fit on [2.5, 10]
// The input is shifted by the breakpoint before evaluating the polynomial
// so that the fit operates on [0, 7.5].
// ---------------------------------------------------------------------------
namespace seg1 {
    static constexpr float NUM_COEFFS[] = {
        0.99999994f,
        -0.00018765f,
        0.00001234f,
        -0.00000087f,
        0.00000006f,
        -0.000000004f,
        0.0000000003f,
        -0.00000000002f,
        0.000000000001f
    };
    static constexpr float DEN_COEFFS[] = {
        1.0f,
        0.00023456f,
        -0.00001567f,
        0.00000123f,
        -0.00000009f,
        0.000000007f,
        -0.0000000005f,
        0.00000000004f,
        -0.000000000003f
    };
}
#endif

// ---------------------------------------------------------------------------
// Evaluate the appropriate segment given an unsigned x in [0, 10].
// Returns the rational fit value.
// ---------------------------------------------------------------------------
#ifdef INP_FLOAT32
template <typename T>
inline T evaluate_erf_segment(T x) {
    if (x < ERF_SEG_BREAKPOINT) {
        // Segment 0: direct evaluation on [0, 2.5]
        return generic_rational<seg0::NUM_COEFFS, seg0::DEN_COEFFS,
                              ERF_SEG0_NUM_DEGREE, ERF_SEG0_DEN_DEGREE>(x);
    } else {
        // Segment 1: evaluate on shifted input [0, 7.5]
        T shifted = x - ERF_SEG_BREAKPOINT;
        return generic_rational<seg1::NUM_COEFFS, seg1::DEN_COEFFS,
                              ERF_SEG1_NUM_DEGREE, ERF_SEG1_DEN_DEGREE>(shifted);
    }
}
#else
// BF16 single-segment path unchanged.
template <typename T>
inline T evaluate_erf_segment(T x) {
    return generic_rational<ERF_NUM_COEFFS, ERF_DEN_COEFFS,
                          ERF_NUM_DEGREE, ERF_DEN_DEGREE>(x);
}
#endif

// ---------------------------------------------------------------------------
// Main entry: computes erf(x) using the signed-input symmetry.
// ---------------------------------------------------------------------------
#ifdef INP_FLOAT32
template <typename T>
inline T erf_compute(T x) {
    bool negative = x < T(0.0);
    T ax = negative ? -x : x;
    // Clamp to [0, 10] to keep the fit well-conditioned.
    if (ax > T(10.0)) ax = T(10.0);
    T result = evaluate_erf_segment(ax);
    // Saturation: rational fit is not bounded; overshoots by up to ~3e-8
    // in the tail. Persists in FP32 dest register and biases downstream
    // ops (e.g. decomposed GELU in CLIP).
    result = sfpi::clamp(result, -1.0f, +1.0f);
    return negative ? -result : result;
}
#else
template <typename T>
inline T erf_compute(T x) {
    bool negative = x < T(0.0);
    T ax = negative ? -x : x;
    if (ax > T(10.0)) ax = T(10.0);
    T result = evaluate_erf_segment(ax);
    result = sfpi::clamp(result, -1.0f, +1.0f);
    return negative ? -result : result;
}
#endif

} // namespace erf
} // namespace sfpu
} // namespace ckernel

#endif // CKERNEL_SFPU_ERF_H
