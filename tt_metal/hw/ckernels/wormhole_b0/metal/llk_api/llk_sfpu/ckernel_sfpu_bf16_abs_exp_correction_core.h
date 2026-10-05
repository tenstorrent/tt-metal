// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Single-row residual replay. Callers own numerical init, replay recording,
// row count, address modes and counter restoration.
namespace sfpi {
constexpr float correction_exp_scale(int degree) {
    float scale = 1.0f;
    for (int i = 0; i < degree; i++) {
        scale *= 0x1p-23f;
    }
    return scale;
}

// Exponent leaf shared by the exp_hw_eval compositions. MULT may be an
// immediate, a programmable constant or a DST value.
template <
    uint32_t DEG,
    bool SquareStore,
    bool SquareFold,
    bool ResidualPool,
    bool ResidualFold,
    bool SkipClamp,
    bool SplitScaleBias,
    typename Mult>
inline vFloat correction_exp_leaf(vFloat x, Mult MULT, float bias, const float* c) {
    vFloat xlog2;
    if constexpr (SplitScaleBias) {
        // An explicit MAD(product, 1, bias) keeps the compiler from contracting the MUL into it.
        vFloat multiplier = MULT;
        xlog2 = __builtin_rvtt_sfpmul(x.get(), multiplier.get(), SFPMAD_MOD1_OFFSET_NONE);
        vFloat one = 1.0f;
        vFloat addend = bias;
        xlog2 = __builtin_rvtt_sfpmad(xlog2.get(), one.get(), addend.get(), SFPMAD_MOD1_OFFSET_NONE);
    } else {
        xlog2 = x * MULT + bias;
    }

    // Full-range safety clamp: keep xlog2 in [0, 255] so the implicit float->int
    // conversion below cannot wrap (TTNN does this in its non-unsafe path).
    vFloat thr_lo = 0.0f;
    vFloat thr_hi = 255.0f;
    if constexpr (!SkipClamp) {
        ordered_min_max(thr_lo, xlog2);  // xlog2 = max(0, xlog2)
        ordered_min_max(xlog2, thr_hi);  // xlog2 = min(xlog2, 255)
    }

    // Branch-free float->int: shift mantissa left by (exp - bias) bits.
    vInt e = exexp(xlog2);
    vInt m = exman(xlog2, MantissaMode::ImplicitOne);
    m = shft(m, e, ShiftMode::Logical);
    vFloat z = as<vFloat>(m);

    vInt ep = exexp(z, ExponentMode::Biased);  // 2^(integer part)
    vMag fm = exman(z);                        // fraction * 2^23
    // Normalize the exman 2^23-scaled fraction back to a float f in [0,1).
    vFloat f;
    if constexpr (SquareFold || ResidualFold) {
        f = convert<vFloat>(fm, RoundMode::Nearest);
    } else {
        f = convert<vFloat>(fm, RoundMode::Nearest) * 0x1p-23f;
    }

    // Plain degree-N Horner for 2^f over the natural [0,1) coeffs.

    vFloat p;
    if constexpr (SquareStore) {
        p = dst_reg[64 + DEG].mode<DataLayout::F32>();
    } else if constexpr (SquareFold) {
        p = c[DEG] * correction_exp_scale(DEG);
    } else if constexpr (ResidualPool) {
        p = vConstFloatPrgm1;
    } else {
        p = c[DEG];
    }
#pragma GCC unroll 16
    for (int k = (int)DEG - 1; k >= 0; k--) {
        if constexpr (SquareStore) {
            if (k == 0) {
                p = p * f + c[0];
            } else {
                p = p * f + vFloat(dst_reg[64 + k].mode<DataLayout::F32>());
            }
        } else if constexpr (SquareFold) {
            p = p * f + c[k] * correction_exp_scale(k);
        } else if constexpr (ResidualPool) {
            if (k == (int)DEG - 1) {
                p = p * f + vConstFloatPrgm2;
            } else {
                p = p * f + c[k];
            }
        } else {
            p = p * f + c[k];
        }
    }

    // Recombine 2^i * 2^f. `ep` is the biased exponent of the integer part
    // (== i + 127). setexp only REPLACES p's exponent field, keeping its
    // mantissa — which is correct only when p in [1,2) (exponent field 127).
    // A polynomial fitted on [0,1) can have g(0)=c[0] just below 1.0 for some
    // degrees (e.g. odd-degree exp2: c0=0.99992), putting p in [0.5,1) at f~=0
    // (exponent field 126). Replacing that with ep then over-scales by 2x. Add
    // p's own exponent deviation from the bias so the integer part composes with
    // p's actual magnitude (mirrors pow_hw_eval's setexp(s, s_exp + q)).
    if constexpr (SquareFold || ResidualFold) {
        return setexp(p, ep);
    } else {
        vInt pe = exexp(p, ExponentMode::Biased);
        return setexp(p, ep + pe - 127);
    }
}

template <typename Config>
inline void abs_exp_park_coefficients() {}

template <uint32_t NumDegree, uint32_t DenDegree, typename Config, typename Exp, typename Reciprocal>
inline __attribute__((always_inline)) vFloat abs_residual_correction(vFloat x, Exp exp, Reciprocal reciprocal) {
    vFloat t;
    vFloat coordinate = setsgn(x, 0);
    vFloat coordinate_bound = __builtin_bit_cast(float, Config::kBoundBits);
    ordered_min_max(coordinate, coordinate_bound);
    t = exp(coordinate);
    vFloat quotient;
    quotient = Config::kNumerator[NumDegree];
#pragma GCC unroll 8
    for (int k = (int)NumDegree - 1; k >= 0; k--) {
        quotient = quotient * t + Config::kNumerator[k];
    }
    vFloat residual = t * quotient;
    vFloat zero = 0.0f;
    vFloat affine = x;
    ordered_min_max(zero, affine);
    return affine + residual;
}

}  // namespace sfpi
