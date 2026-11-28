// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Single-row residual replay. Callers own numerical init, replay recording,
// row count, address modes and counter restoration.
namespace ckernel::sfpu::ttpoly::erfc {
using ::sfpi::as;
using ::sfpi::convert;
using ::sfpi::DataLayout;
using ::sfpi::exexp;
using ::sfpi::exman;
using ::sfpi::ExponentMode;
using ::sfpi::MantissaMode;
using ::sfpi::RoundMode;
using ::sfpi::setexp;
using ::sfpi::setsgn;
using ::sfpi::SFPMAD_MOD1_OFFSET_NONE;
using ::sfpi::shft;
using ::sfpi::ShiftMode;
using ::sfpi::vFloat;
using ::sfpi::vInt;
using ::sfpi::vMag;
template <
    uint32_t DEG,
    bool SquareStore,
    bool SquareFold,
    bool ResidualPool,
    bool ResidualFold,
    bool SkipClamp,
    bool SplitScaleBias,
    typename Mult>
__attribute__((always_inline)) inline vFloat correction_exp_leaf(vFloat x, Mult MULT, float bias, const float* c) {
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

    // Branch-free float->int: shift mantissa left by (exp - bias) bits.
    vInt e = exexp(xlog2);
    vInt m = exman(xlog2, MantissaMode::ImplicitOne);
    m = shft(m, e, ShiftMode::Logical);
    vFloat z = as<vFloat>(m);

    vInt ep = exexp(z, ExponentMode::Biased);  // 2^(integer part)
    vMag fm = exman(z);                        // fraction * 2^23
    // Normalize the exman 2^23-scaled fraction back to a float f in [0,1).
    vFloat f;
    f = convert<vFloat>(fm, RoundMode::Nearest) * 0x1p-23f;

    // Plain degree-N Horner for 2^f over the natural [0,1) coeffs.

    vFloat p;
    p = c[DEG];
#pragma GCC unroll 16
    for (int k = (int)DEG - 1; k >= 0; k--) {
        p = p * f + c[k];
    }

    {
        vInt pe = exexp(p, ExponentMode::Biased);
        return setexp(p, ep + pe - 127);
    }
}

template <typename Config>
inline void abs_exp_park_coefficients() {}

template <uint32_t NumDegree, uint32_t DenDegree, typename Config, typename Exp, typename Reciprocal>
inline __attribute__((always_inline)) vFloat abs_square_correction(vFloat x, Exp exp, Reciprocal reciprocal) {
    vFloat coordinate = setsgn(x, 0);
    {
        vFloat coordinate_bound = __builtin_bit_cast(float, Config::kBoundBits);
        ordered_min_max(coordinate, coordinate_bound);
    }
    vFloat negative_square = -(coordinate * coordinate);
    vFloat decay = exp(negative_square);
    vFloat quotient;
    {
        vFloat num;
        vFloat den;
        num = Config::kNumerator[NumDegree];
        den = Config::kDenominator[DenDegree];
#pragma GCC unroll 8
        for (int k = (int)NumDegree - 1; k >= 0; k--) {
            num = num * coordinate + Config::kNumerator[k];
        }
#pragma GCC unroll 8
        for (int k = (int)DenDegree - 1; k >= 0; k--) {
            den = den * coordinate + Config::kDenominator[k];
        }
        quotient = num * reciprocal(den);
    }
    vFloat result = decay * quotient;
    v_if(x < 0.0f) { result = 2.0f - result; }
    v_endif;
    return result;
}
}  // namespace ckernel::sfpu::ttpoly::erfc
