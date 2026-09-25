// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Exact selected single-row residual replay. Callers own numerical init,
// replay recording, row count, address modes and counter restoration.
namespace sfpi {
constexpr float correction_exp_scale(int degree) {
    float scale = 1.0f;
    for (int i = 0; i < degree; i++) {
        scale *= 0x1p-23f;
    }
    return scale;
}

// Ordinary exponent leaf shared with all original exp_hw_eval compositions.
// MULT may be an immediate or the selected programmable/DST value.
template <
    uint32_t DEG,
    bool SquareStore,
    bool SquareFold,
    bool ResidualPool,
    bool ResidualFold,
    bool SkipClamp,
    typename Mult>
inline vFloat correction_exp_leaf(vFloat x, Mult MULT, float bias, const float* c) {
    vFloat xlog2 = x * MULT + bias;

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
                if constexpr (ResidualFold) {
                    p = p * f + vFloat(dst_reg[68].mode<DataLayout::F32>());
                } else {
                    p = p * f + c[k];
                }
            }
        } else {
            p = p * f + c[k];
        }
    }

    // Recombine 2^i * 2^f. `ep` is the biased exponent of the integer part
    // (== i + 127). setexp only REPLACES p's exponent field, keeping its
    // mantissa — which is correct only when p in [1,2) (exponent field 127).
    // The fitter's natural [0,1) fit makes g(0)=c[0] dip just below 1.0 for some
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
inline void abs_exp_park_coefficients() {
    if constexpr (Config::kCorrectionStore) {
        if constexpr ((__builtin_bit_cast(uint32_t, Config::kNumerator[0]) & 0xffffu) != 0u) {
            dst_reg[64].mode<DataLayout::F32>() = vFloat(Config::kNumerator[0]);
        }
        if constexpr ((__builtin_bit_cast(uint32_t, Config::kNumerator[1]) & 0xffffu) != 0u) {
            dst_reg[65].mode<DataLayout::F32>() = vFloat(Config::kNumerator[1]);
        }
        if constexpr ((__builtin_bit_cast(uint32_t, Config::kNumerator[2]) & 0xffffu) != 0u) {
            dst_reg[66].mode<DataLayout::F32>() = vFloat(Config::kNumerator[2]);
        }
        if constexpr ((__builtin_bit_cast(uint32_t, Config::kNumerator[3]) & 0xffffu) != 0u) {
            dst_reg[67].mode<DataLayout::F32>() = vFloat(Config::kNumerator[3]);
        }
    }
    if constexpr (Config::kResidualFold) {
        dst_reg[68].mode<DataLayout::F32>() = vFloat(Config::kExpCoefficients[0]);
    }
    if constexpr (Config::kSquareStore) {
        dst_reg[64].mode<DataLayout::F32>() = vFloat(Config::kMultiplier);
#pragma GCC unroll 3
        for (int k = 1; k <= 3; ++k) {
            dst_reg[64 + k].mode<DataLayout::F32>() = vFloat(Config::kExpCoefficients[k] * correction_exp_scale(k));
        }
        dst_reg[68].mode<DataLayout::F32>() = vFloat(Config::kNumerator[0]);
        dst_reg[69].mode<DataLayout::F32>() = vFloat(Config::kNumerator[1]);
        dst_reg[70].mode<DataLayout::F32>() = vFloat(Config::kDenominator[0]);
        dst_reg[71].mode<DataLayout::F32>() = vFloat(Config::kDenominator[2]);
    }
}

template <typename Config, uint32_t INDEX>
inline auto correction_coefficient() {
    constexpr uint32_t bits = __builtin_bit_cast(uint32_t, Config::kNumerator[INDEX]);
    if constexpr ((bits & 0xffffu) != 0u) {
        return vFloat(dst_reg[64u + INDEX].mode<DataLayout::F32>());
    } else {
        return Config::kNumerator[INDEX];
    }
}

template <uint32_t NumDegree, uint32_t DenDegree, typename Config, typename Exp, typename Reciprocal>
inline vFloat abs_residual_correction(vFloat x, Exp exp, Reciprocal reciprocal) {
    vFloat t;
    if constexpr (Config::kHasBound) {
        vFloat coordinate = setsgn(x, 0);
        vFloat coordinate_bound = __builtin_bit_cast(float, Config::kBoundBits);
        ordered_min_max(coordinate, coordinate_bound);
        t = exp(coordinate);
    } else {
        t = exp(x);
    }
    vFloat quotient;
    if constexpr (Config::kPolynomial) {
        if constexpr (Config::kCorrectionStore) {
            static_assert(NumDegree == 3u);
            quotient = correction_coefficient<Config, 3>();
            quotient = quotient * t + correction_coefficient<Config, 2>();
            quotient = quotient * t + correction_coefficient<Config, 1>();
            quotient = quotient * t + correction_coefficient<Config, 0>();
        } else {
            quotient = Config::kNumerator[NumDegree];
#pragma GCC unroll 8
            for (int k = (int)NumDegree - 1; k >= 0; k--) {
                quotient = quotient * t + Config::kNumerator[k];
            }
        }
    } else {
        vFloat num = Config::kNumerator[NumDegree];
#pragma GCC unroll 8
        for (int k = (int)NumDegree - 1; k >= 0; k--) {
            num = num * t + Config::kNumerator[k];
        }
        vFloat den = Config::kDenominator[DenDegree];
#pragma GCC unroll 8
        for (int k = (int)DenDegree - 1; k >= 0; k--) {
            den = den * t + Config::kDenominator[k];
        }
        quotient = num * reciprocal(den);
    }
    vFloat residual = t * quotient;
    vFloat zero = 0.0f;
    vFloat affine = x;
    if constexpr (Config::kPositivePart) {
        ordered_min_max(zero, affine);
        return affine + residual;
    } else {
        ordered_min_max(affine, zero);
        return affine - residual;
    }
}

template <uint32_t NumDegree, uint32_t DenDegree, typename Config, typename Exp, typename Reciprocal>
inline vFloat abs_square_correction(vFloat x, Exp exp, Reciprocal reciprocal) {
    vFloat coordinate = setsgn(x, 0);
    if constexpr (Config::kHasBound) {
        vFloat coordinate_bound = __builtin_bit_cast(float, Config::kBoundBits);
        ordered_min_max(coordinate, coordinate_bound);
    }
    vFloat negative_square = -(coordinate * coordinate);
    vFloat decay = exp(negative_square);
    vFloat quotient;
    if constexpr (Config::kPolynomial) {
        quotient = Config::kNumerator[NumDegree];
#pragma GCC unroll 8
        for (int k = (int)NumDegree - 1; k >= 0; k--) {
            quotient = quotient * coordinate + Config::kNumerator[k];
        }
    } else {
        vFloat num;
        vFloat den;
        if constexpr (Config::kSquareStore) {
            static_assert(NumDegree == 1 && DenDegree == 2);
            num = dst_reg[69].mode<DataLayout::F32>();
            den = dst_reg[71].mode<DataLayout::F32>();
        } else {
            num = Config::kNumerator[NumDegree];
            den = Config::kDenominator[DenDegree];
        }
#pragma GCC unroll 8
        for (int k = (int)NumDegree - 1; k >= 0; k--) {
            if constexpr (Config::kSquareStore) {
                num = num * coordinate + vFloat(dst_reg[68 + k].mode<DataLayout::F32>());
            } else {
                num = num * coordinate + Config::kNumerator[k];
            }
        }
#pragma GCC unroll 8
        for (int k = (int)DenDegree - 1; k >= 0; k--) {
            if constexpr (Config::kSquareStore) {
                if (k == 0) {
                    den = den * coordinate + vFloat(dst_reg[70].mode<DataLayout::F32>());
                } else {
                    den = den * coordinate + Config::kDenominator[k];
                }
            } else {
                den = den * coordinate + Config::kDenominator[k];
            }
        }
        quotient = num * reciprocal(den);
    }
    vFloat result = decay * quotient;
    v_if(x < 0.0f) { result = 2.0f - result; }
    v_endif;
    return result;
}

#if defined(ARCH_BLACKHOLE)
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

template <typename Config, uint32_t Hold>
inline void abs_exp_residual_core() {
    constexpr uint32_t kCorrectionC0 = __builtin_bit_cast(uint32_t, Config::kNumerator[0]);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, Hold, 0);
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 1);  // |x|
    if constexpr (Config::kHasBound) {
        TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, (Config::kBoundBits >> 16));
        TTI_SFPSWAP(0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0,
                    1);  // coordinate=min(|x|, bound)
    } else {
        // The same typed form may own the complete exponent range without a
        // declared coordinate terminal.  Preserve the two producer/consumer
        // issue slots; substituting a clamp here would change its finite tail.
        TTI_SFPNOP;
        TTI_SFPNOP;
    }
    // Match the compiler body exactly: its immediate exponent bias lowers to
    // separate MUL then ADDI, not a fused MAD.  Fusing these two source
    // operations changes one BF16 output lane after final rounding.
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG12,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG0,
        0);      // xlog2=coordinate*(-log2e)
    TTI_SFPNOP;  // replay bypasses the MUL -> ADDI scoreboard stall
    TTI_SFPADDI(0x42fe, ckernel::p_sfpu::LREG0,
                0);  // xlog2 += exact 127.0 exponent bias
    TTI_SFPNOP;
    TTI_SFPEXEXP(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1, 0);
    TTI_SFPEXMAN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG0, 0);
    TTI_SFPSHFT(0, 1, 0, 0);
    TTI_SFPEXMAN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1, sfpi::SFPEXMAN_MOD1_PAD9);
    TTI_SFPCAST(ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG1, 0);
    TTI_SFPMAD(ckernel::p_sfpu::LREG13, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG14, ckernel::p_sfpu::LREG2, 0);
    TTI_SFPLOAD(
        ckernel::p_sfpu::LREG3,
        sfpi::SFPLOAD_MOD0_FMT_UINT16,
        Hold,
        0);  // raw BH DST word, fills the exponent Horner hazard slot
    TTI_SFPMAD(ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG7, ckernel::p_sfpu::LREG1, 0);
    TTI_SFPNOP;  // replay bypasses the MAD -> SETEXP scoreboard stall
    // The fitted exponent P2 is certified in [1,2), so the ordinary bare
    // SETEXP reconstruction is exact and leaves three replay slots for the
    // shared target-class finalizer below.
    TTI_SFPSETEXP(
        0,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG0,
        2);  // t=p*2^i, exponent source is the retained z bits in L0
    TTI_SFPMAD(ckernel::p_sfpu::LREG4, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG6, ckernel::p_sfpu::LREG2, 0);
    TTI_SFPNOP;
    TTI_SFPMAD(ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG2, 0);
    if constexpr (Config::kNumerator[0] == 1.0f) {
        TTI_SFPNOP;
    } else {
        TTI_SFPLOADI(ckernel::p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, (kCorrectionC0 >> 16));
        TTI_SFPLOADI(ckernel::p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, (kCorrectionC0 & 0xffffu));
    }
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG0,
        (Config::kNumerator[0] == 1.0f) ? ckernel::p_sfpu::LCONST_1 : ckernel::p_sfpu::LREG7,
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
    if constexpr (Config::kPositivePart) {
        TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG1,
                    9);  // affine=max(x,0)
    } else {
        TTI_SFPSWAP(0, ckernel::p_sfpu::LCONST_0, ckernel::p_sfpu::LREG1,
                    1);  // affine=min(x,0)
    }
    if constexpr (Config::kPositivePart) {
        TTI_SFPADD(
            ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG0, 0);
    } else {
        TTI_SFPMAD(
            ckernel::p_sfpu::LCONST_neg1, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG0, 0);
    }
}

template <typename Config, uint32_t Advance>
inline void abs_exp_residual_suffix() {
    static_assert(Config::kNegativeInfinityClass >= 1 && Config::kNegativeInfinityClass <= 3);
    constexpr uint32_t kExpC0 = __builtin_bit_cast(uint32_t, Config::kExpCoefficients[0]);
    if constexpr (Config::kNegativeInfinityClass == 3) {
        TTI_SFPMUL(
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LCONST_0,
            ckernel::p_sfpu::LREG1,
            0);  // typed -Inf quotient terminal is +0
    } else if constexpr (Config::kNegativeInfinityClass == 2) {
        TTI_SFPSETEXP(255, ckernel::p_sfpu::LCONST_neg1, ckernel::p_sfpu::LREG1,
                      1);  // exact -Inf override value
    } else if constexpr (Config::kNegativeInfinityClass == 1) {
        TTI_SFPSETEXP(255, ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG1,
                      1);  // exact +Inf override value
    }
    TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_USHORT, 0x80ff);
    TTI_SFPXOR(0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG3, 0);
    TTI_SFPAND(ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG3, ckernel::p_sfpu::LREG2, 1);
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG2, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG3, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPMOV(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG0, 0);
    TTI_SFPENCC(0, 0, 0, 0);
    TTI_SFP_STOCH_RND(
        sfpi::SFPSTOCHRND_RND_EVEN,
        0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG0,
        sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPSTORE(ckernel::p_sfpu::LREG0, 0, Advance, 0);
    if constexpr (Config::kNumerator[0] != 1.0f) {
        // The non-unit correction origin uses LREG7 after SETEXP, while
        // the next recorded exponent body expects its pinned exp c0
        // there.  Restore that schedule-owned cross-row resource.
        TTI_SFPLOADI(ckernel::p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, (kExpC0 >> 16));
        TTI_SFPLOADI(ckernel::p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, (kExpC0 & 0xffffu));
    }
}
#endif
}  // namespace sfpi
