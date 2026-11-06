// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Single-row residual replay. Callers own numerical init, replay recording,
// row count, address modes and counter restoration.
namespace ckernel::sfpu::ttpoly::logsigmoid {
using namespace ::sfpi;
using ::sfpi::DataLayout;
constexpr float correction_exp_scale(int degree) {
    float scale = 1.0f;
    for (int i = 0; i < degree; i++) {
        scale *= 0x1p-23f;
    }
    return scale;
}

template <
    uint32_t DEG,
    bool SquareStore,
    bool SquareFold,
    bool ResidualPool,
    bool ResidualFold,
    bool SkipClamp,
    bool SplitScaleBias,
    typename Mult,
    bool PoolHeld = false>
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
    } else if constexpr (ResidualPool && PoolHeld) {
        p = ResidualFold ? c[DEG] * correction_exp_scale(DEG) : c[DEG];
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
        } else if constexpr (ResidualPool && PoolHeld) {
            if (k == (int)DEG - 1) {
                p = p * f + (ResidualFold ? c[k] * correction_exp_scale(k) : c[k]);
            } else {
                p = p * f + c[k];
            }
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
inline __attribute__((always_inline)) vFloat
abs_residual_correction(vFloat x, Exp exp, [[maybe_unused]] Reciprocal reciprocal) {
    vFloat t;
    {
        vFloat coordinate = setsgn(x, 0);
        vFloat coordinate_bound = __builtin_bit_cast(float, Config::kBoundBits);
        ordered_min_max(coordinate, coordinate_bound);
        t = exp(coordinate);
    }
    vFloat quotient;
    quotient = Config::kNumerator[NumDegree];
#pragma GCC unroll 8
    for (int k = (int)NumDegree - 1; k >= 0; k--) {
        quotient = quotient * t + Config::kNumerator[k];
    }
    vFloat residual = t * quotient;
    vFloat zero = 0.0f;
    vFloat affine = x;
    ordered_min_max(affine, zero);
    return affine - residual;
}

template <typename Config>
inline void abs_exp_residual_pins() {
    constexpr uint32_t bits4 = __builtin_bit_cast(uint32_t, Config::kNumerator[3]);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_UPPER, bits4 >> 16);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG4, sfpi::SFPLOADI_MOD0_LOWER, bits4 & 0xffffu);
    constexpr uint32_t bits5 = __builtin_bit_cast(uint32_t, Config::kNumerator[1]);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_UPPER, bits5 >> 16);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_LOWER, bits5 & 0xffffu);
    constexpr uint32_t bits6 = __builtin_bit_cast(uint32_t, Config::kNumerator[2]);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER, bits6 >> 16);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, bits6 & 0xffffu);
    constexpr uint32_t bits7 = __builtin_bit_cast(uint32_t, Config::kExpCoefficients[0]);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, bits7 >> 16);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, bits7 & 0xffffu);
}

// The exponent leaf's coefficient of degree kExpDegree - Below, pre-scaled as init_abs_exp_correction
// programs it in Prgm1 (Below 0) or Prgm2 (Below 1).
template <typename Config, uint32_t Below>
constexpr uint32_t abs_exp_held_coefficient() {
    constexpr uint32_t degree = Config::kExpDegree - Below;
    return __builtin_bit_cast(uint32_t, Config::kExpCoefficients[degree] * correction_exp_scale(degree));
}

// Held loads the multiplier and the two upper exponent coefficients into L3 and L2 per row, in
// slots the Prgm0..2 form leaves as NOPs and two more, instead of reading them from Prgm0..2.
// The slots abs_exp_residual_core_head issues: the part of the row a window too short for all of it records.
template <typename Config, uint32_t Hold, bool Held = false>
inline constexpr uint32_t abs_exp_residual_core_head_slots = 13u + ((Held) ? 2u : 0u);
template <typename Config, uint32_t Hold, bool Held = false>
inline void abs_exp_residual_core_head() {
    constexpr uint32_t kMultiplierReg = Held ? ckernel::p_sfpu::LREG3 : ckernel::p_sfpu::LREG12;
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, Hold, 0);
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 1);  // |x|
    if constexpr (Held) {
        constexpr uint32_t kMultiplierBits = __builtin_bit_cast(uint32_t, Config::kMultiplier);
        TTI_SFPLOADI(ckernel::p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, kMultiplierBits >> 16);
        TTI_SFPLOADI(ckernel::p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, kMultiplierBits & 0xffffu);
    }
    TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, (Config::kBoundBits >> 16));
    TTI_SFPSWAP(0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0,
                1);  // coordinate=min(|x|, bound)
    // A separate MUL then ADDI, not a fused MAD: fusing the exponent bias into
    // the multiply changes one BF16 output after final rounding.
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG0,
        kMultiplierReg,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG0,
        0);  // xlog2=coordinate*(-log2e)
    // replay bypasses the MUL -> ADDI scoreboard stall
    if constexpr (Held) {
        TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_UPPER, (abs_exp_held_coefficient<Config, 0>() >> 16));
    } else {
        TTI_SFPNOP;
    }
    TTI_SFPADDI(0x42fe, ckernel::p_sfpu::LREG0,
                0);  // xlog2 += exact 127.0 exponent bias
    if constexpr (Held) {
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_LOWER, (abs_exp_held_coefficient<Config, 0>() & 0xffffu));
    } else {
        TTI_SFPNOP;
    }
    TTI_SFPEXEXP(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1, 0);
    TTI_SFPEXMAN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 0);
    TTI_SFPSHFT(0, 1, 0, 0);
    TTI_SFPEXMAN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1, sfpi::SFPEXMAN_MOD1_PAD9);
    TTI_SFPCAST(ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG1, 0);
}
template <typename Config, uint32_t Hold, bool Held = false>
inline void abs_exp_residual_core_tail() {
    constexpr uint32_t kCorrectionC0 = __builtin_bit_cast(uint32_t, Config::kNumerator[0]);
    constexpr uint32_t kC2Reg = Held ? ckernel::p_sfpu::LREG2 : ckernel::p_sfpu::LREG13;
    constexpr uint32_t kC1Reg = Held ? ckernel::p_sfpu::LREG3 : ckernel::p_sfpu::LREG14;
    if constexpr (Held) {
        TTI_SFPLOADI(ckernel::p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, (abs_exp_held_coefficient<Config, 1>() >> 16));
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, (abs_exp_held_coefficient<Config, 1>() & 0xffffu));
    }
    TTI_SFPMAD(kC2Reg, ckernel::p_sfpu::LREG1, kC1Reg, ckernel::p_sfpu::LREG2, 0);
    TTI_SFPLOAD(
        ckernel::p_sfpu::LREG3,
        sfpi::SFPLOAD_MOD0_FMT_UINT16,
        Hold,
        0);  // raw BH DST word, fills the exponent Horner hazard slot
    TTI_SFPMAD(ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG1, 0);
    TTI_SFPNOP;  // replay bypasses the MAD -> SETEXP scoreboard stall
    // The exponent polynomial P2 stays in [1,2) over its whole input range, so
    // a bare SETEXP reconstruction is exact and leaves three replay slots for
    // the class finalizer below.
    TTI_SFPSETEXP(
        0,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG0,
        2);  // t=p*2^i, exponent source is the retained z bits in L0
    TTI_SFPMAD(ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG2, 0);
    TTI_SFPNOP;
    TTI_SFPMAD(ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG2, 0);
    if constexpr (__builtin_bit_cast(uint32_t, Config::kNumerator[0]) == 0x3f800000u) {
        TTI_SFPNOP;
    } else {
        TTI_SFPLOADI(ckernel::p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, (kCorrectionC0 >> 16));
        TTI_SFPLOADI(ckernel::p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, (kCorrectionC0 & 0xffffu));
    }
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG0,
        (__builtin_bit_cast(uint32_t, Config::kNumerator[0]) == 0x3f800000u) ? ckernel::p_sfpu::LCONST_1
                                                                             : ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG2,
        0);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG1, 0, Hold,
                0);  // raw fp lane, fills the final correction hazard slot
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG2,
        0);  // residual=t*Q(t)
    TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG1,
                1);  // affine=min(x,0)
    TTI_SFPMAD(ckernel::p_sfpu::LCONST_neg1, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG0, 0);
}

template <typename Config, uint32_t Advance>
inline void abs_exp_residual_suffix() {
    constexpr uint32_t kExpC0 = __builtin_bit_cast(uint32_t, Config::kExpCoefficients[0]);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_USHORT, 0x00ff);
    TTI_SFPXOR(0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG3, 0);
    TTI_SFPAND(ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG2, 1);
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG2, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_USHORT, 0x7f00);
    TTI_SFPAND(ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG3, 1);
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG3, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, 0x7f80);  // +Inf
    TTI_SFPENCC(0, 0, 0, 0);
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN,
        0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPSTORE(ckernel::p_sfpu::LREG0, 0, Advance, 0);
    if constexpr (__builtin_bit_cast(uint32_t, Config::kNumerator[0]) != 0x3f800000u) {
        TTI_SFPLOADI(ckernel::p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, (kExpC0 >> 16));
        TTI_SFPLOADI(ckernel::p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, (kExpC0 & 0xffffu));
    }
}
}  // namespace ckernel::sfpu::ttpoly::logsigmoid
