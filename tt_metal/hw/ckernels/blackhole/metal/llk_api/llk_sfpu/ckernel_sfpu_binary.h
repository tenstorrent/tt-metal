// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_conversions.h"
#include "ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_log.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

sfpi_inline sfpi::vFloat calculate_sfpu_binary_power(sfpi::vFloat base, sfpi::vFloat pow) {
    sfpi::vFloat original_base = base;

    // Check for integer power
    sfpi::vSMag16 pow_smag = sfpi::convert<sfpi::vSMag16>(
        pow, sfpi::RoundMode::Nearest);  // int16 should be plenty, since large powers will approach 0/Inf
    sfpi::vFloat pow_rounded = sfpi::convert<sfpi::vFloat>(pow_smag, sfpi::RoundMode::Nearest);
    v_if(pow_rounded == pow) {
        // if pow is integer, set base to positive
        base = sfpi::setsgn(base, 0);
    }
    v_endif;

    // Normalize base to calculation range
    sfpi::vFloat x = sfpi::setexp(base, 127);  // set exp to exp bias (put base in range of 1-2)

    // 3rd order polynomial approx - determined using rminimax over [1,2], see LogPolyNoInit
    sfpi::vFloat series_result =
        x * (x * (x * LogPolyNoInit::A - LogPolyNoInit::B) + LogPolyNoInit::C) - LogPolyNoInit::D;

    // Convert exponent to float
    sfpi::vSMag exp = sfpi::convert<sfpi::vSMag>(exexp(base));
    sfpi::vFloat expf = sfpi::convert<sfpi::vFloat>(exp, sfpi::RoundMode::Nearest);

    // De-normalize to original range
    sfpi::vFloat vConstLn2 = LogPolyNoInit::LN2;
    sfpi::vFloat log_result = expf * vConstLn2 + series_result;  // exp correction: ln(1+x) + exp*ln(2)

    // Base case when input is 0. ln(0) = -inf
    v_if(base == 0.0f) {  // Reload for register pressure
        log_result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    // Take exp(pow * log(base)) to produce base^pow
    sfpi::vFloat val = pow * log_result;

    // Force sign to 0 (make number positive)
    sfpi::vFloat result = _sfpu_exp_(sfpi::setsgn(val, 0));

    v_if(val < 0) { result = sfpu_reciprocal_iter<2>(result); }
    v_endif;

    // Check valid base range
    v_if(original_base < 0.0f) {  // negative base
        // Check for integer power
        v_if(pow_rounded == pow) {
            // if pow is odd integer, set result to negative
            // Check if odd by dividing by 2 and comparing with floor
            sfpi::vFloat half_pow = pow_rounded * 0.5f;
            sfpi::vSMag16 half_pow_int = sfpi::convert<sfpi::vSMag16>(half_pow, sfpi::RoundMode::Nearest);
            sfpi::vFloat half_pow_floored = sfpi::convert<sfpi::vFloat>(half_pow_int, sfpi::RoundMode::Nearest);
            v_if(half_pow != half_pow_floored) { result = sfpi::setsgn(result, 1); }
            v_endif;
        }
        v_else { result = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
    }
    v_endif;

    // IEEE 754: pow(x, 0) == 1 for every x, including 0, +/-inf and NaN. Without this the
    // composition above forms 0 * ln(0) = 0 * -inf = NaN at base == 0 (SFPMAD), exp(NaN)
    // collapses to +0, and the v_if(val < 0) is then evaluated on a NaN, which the ISA
    // leaves undefined (VectorUnit, SFPSETCC) -- measured on Wormhole as 0**0 = 0 but
    // 0**-0.0 = inf, and on Blackhole as inf for both, the same predicate resolving one way
    // there instead of two.
    // Last, so the negative-base sign flip above cannot turn (-2)**0 into -1. Compared on
    // setsgn(pow, 0) because SFPSETCC's contract excludes negative zero: measured, a bare
    // pow == 0.0f does not fire for pow == -0.0 and leaves 0**-0.0 at inf.
    v_if(sfpi::setsgn(pow, 0) == 0.0f) { result = 1.0f; }
    v_endif;

    return result;
}

template <
    bool APPROXIMATION_MODE,
    BinaryOp BINOP,
    int ITERATIONS,
    bool is_fp32_dest_acc_en,
    DstRoundingMode dst_rounding_mode = DstRoundingMode::Default>
inline void calculate_sfpu_binary(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    // XLOGY: the log body's two polynomial constants are bound here and held in LREGs across the
    // loop; as literals inside the loop they would be re-materialised on every row.
    // Declared for every op but loaded only for XLOGY: sfpi does not drop an unused SFPLOADI.
    // Unassigned for every other op, so do not read them outside the XLOGY branch.
    sfpi::vFloat log_c;
    sfpi::vFloat log_d;
    if constexpr (BINOP == BinaryOp::XLOGY) {
        log_c = LogPoly::C;
        log_d = LogPoly::D;
    }
    // SFPU microcode, one row of a face
    auto row = [&]() __attribute__((always_inline)) {
        // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
        constexpr std::uint32_t dst_tile_size_sfpi = 32;
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat result = 0.0f;

        if constexpr (BINOP == BinaryOp::ADD) {
            result = in0 + in1;
        } else if constexpr (BINOP == BinaryOp::SUB) {
            result = in0 - in1;
        } else if constexpr (BINOP == BinaryOp::MUL) {
            result = in0 * in1;
        } else if constexpr (BINOP == BinaryOp::DIV) {
            result = in0 * sfpu_reciprocal_iter<2>(in1);
        } else if constexpr (BINOP == BinaryOp::RSUB) {
            result = in1 - in0;
        } else if constexpr (BINOP == BinaryOp::POW) {
            result = calculate_sfpu_binary_power(in0, in1);
        } else if constexpr (BINOP == BinaryOp::XLOGY) {
            v_if((in1 < 0.0f) || (in1 == nan)) { result = nan; }
            v_else {
                sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = in1;
                _calculate_log_body_(log_c, log_d, dst_index_out);
                result = sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] * in0;
            }
            v_endif;
        } else if constexpr (BINOP == BinaryOp::NEXTAFTER || BINOP == BinaryOp::NEXTAFTER_BF16) {
            // Step in0 one representable value toward in1. Consecutive floats of one sign are
            // consecutive integers when the bit pattern is read as an integer, so the step is taken
            // there: that gives one ULP at in0's own magnitude, which a fixed epsilon cannot.
            // bfloat16 keeps its mantissa in the top 16 bits of the fp32 dest register, so one of
            // its ULPs is 0x10000 there.
            constexpr int kUlpStep = (BINOP == BinaryOp::NEXTAFTER_BF16) ? 0x10000 : 1;
            // Kept flat, with every value declared up front: the sfpi predication pass does not
            // survive a v_if nested inside a v_else here.
            sfpi::vInt bits = sfpi::as<sfpi::vInt>(in0);
            // A step of zero leaves in0 alone, which is what in0 == in1 wants.
            sfpi::vInt step = 0;
            // The bit pattern grows away from zero for either sign, so the direction of the step
            // depends on in0's sign as well as on which side in1 lies.
            v_if(in0 < in1 && in0 >= 0.0f) { step = kUlpStep; }
            v_endif;
            v_if(in0 < in1 && in0 < 0.0f) { step = -kUlpStep; }
            v_endif;
            v_if(in0 > in1 && in0 > 0.0f) { step = -kUlpStep; }
            v_endif;
            v_if(in0 > in1 && in0 <= 0.0f) { step = kUlpStep; }
            v_endif;
            result = sfpi::as<sfpi::vFloat>(bits + step);
            // Zeros need their own path: neither zero reaches its neighbour by stepping its own
            // pattern, and the sign of the answer comes from the target rather than from in0.
            // The guards test the magnitudes, not in0 and in1 directly, because SFPSETCC is
            // specified only for a comparand that is not negative zero (VectorUnit.md), and
            // in0 == 0.0f tests in0 - 0.0f, which is negative zero exactly when in0 is. Clearing
            // the sign first is what brings -0.0 into the guard.
            sfpi::vFloat mag_a = sfpi::setsgn(in0, 0);
            sfpi::vFloat mag_b = sfpi::setsgn(in1, 0);
            // vConstFloatPrgm2 holds the bit pattern kUlpStep, programmed by sfpu_binary_init.
            v_if(mag_a == 0.0f && in1 > 0.0f) { result = sfpi::vConstFloatPrgm2; }
            v_endif;
            v_if(mag_a == 0.0f && in1 < 0.0f) { result = -sfpi::vFloat(sfpi::vConstFloatPrgm2); }
            v_endif;
            // Equal operands return the target, so two zeros return in1's zero, which is not
            // always in0's: nextafter(+0, -0) is -0. This runs after the two guards above because
            // they compare in1 against zero, which is itself unspecified when in1 is negative zero.
            v_if(mag_a == 0.0f && mag_b == 0.0f) { result = in1; }
            v_endif;
            // A NaN in either operand has to propagate, and nothing above arranges that: SFPSETCC
            // is specified only for a comparand that is neither negative zero nor NaN, so the
            // direction predicates decide arbitrarily here. Measured on silicon, nextafter(1.0f,
            // NaN) stepped its operand and returned 1.0000001 rather than NaN. Classify on the
            // integer pattern instead -- exponent all ones with a non-zero mantissa -- the same
            // reason ckernel_sfpu_isclose.h reads bit patterns for its own Inf/NaN lanes. A
            // widened bfloat16 NaN has that exponent and a non-zero mantissa too, so this serves
            // both entry points unchanged. Last, so it wins over the direction and zero arms.
            // vConstIntPrgm0 = 0x7FFFFFFF and vConstIntPrgm1 = 0x7F800000, programmed by sfpu_binary_init.
            v_if((bits & sfpi::vConstIntPrgm0) > sfpi::vConstIntPrgm1) { result = nan; }
            v_endif;
            v_if((sfpi::as<sfpi::vInt>(in1) & sfpi::vConstIntPrgm0) > sfpi::vConstIntPrgm1) { result = nan; }
            v_endif;
        }

        if constexpr (
            (BINOP == BinaryOp::ADD || BINOP == BinaryOp::SUB || BINOP == BinaryOp::RSUB) && !is_fp32_dest_acc_en &&
            dst_rounding_mode == DstRoundingMode::NearestEven) {
            result = float32_to_bf16_rne(result);
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    };

    if constexpr (BINOP == BinaryOp::POW) {
        // Not unrolled: the long pow body is slower unrolled.
        for (int d = 0; d < ITERATIONS; d++) {
            row();
        }
    } else {
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            row();
        }
    }
}

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_binary_mul(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr std::uint32_t dst_tile_size_sfpi = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat result = in0 * in1;

        if constexpr (!is_fp32_dest_acc_en) {
            // Software RNE with 0 * x = 0 and x * 0 = 0, to match FPU behaviour for bfloat16 multiplication:
            // where either input is zero the sum stays 0x7fff, which the mask turns into +0.
            sfpi::vUInt bits = sfpi::as<sfpi::vUInt>(result);
            sfpi::vUInt rounded = 0x7fffU;
            v_if(in0 != 0 && in1 != 0) { rounded += bits + ((bits >> 16) & 1); }
            v_endif;
            result = sfpi::as<sfpi::vFloat>(rounded & 0xFFFF0000U);
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_binary_div(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr std::uint32_t dst_tile_size_sfpi = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat result;
        if constexpr (is_fp32_dest_acc_en) {
            sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
            sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

            // Refine signed mantissas with magnitudes in [1, 2), so neither the
            // reciprocal nor the residual underflows, then restore the exponent.
            sfpi::vFloat ma = sfpi::setexp(in0, 127);
            sfpi::vFloat mb = sfpi::setexp(in1, 127);
            // The hardware seed needs one Newton step before quotient refinement.
            sfpi::vFloat r = sfpu_reciprocal_iter<1, true>(mb);
            sfpi::vFloat q = ma * r;
            sfpi::vFloat residual = ma - q * mb;
            q = q + residual * r;

            sfpi::vInt ea = sfpi::exexp(in0, sfpi::ExponentMode::Biased);
            // eb is unbiased, which absorbs the bias in exponent below.
            sfpi::vInt eb = sfpi::exexp(in1);
            // Split exponent restoration between two factors. Their product
            // supplies hardware overflow/underflow handling, while power-of-two
            // scaling is exact for normal results.
            sfpi::vInt exponent = ea - eb + sfpi::exexp(q, sfpi::ExponentMode::Biased);
            sfpi::vInt half = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(exponent) >> 1);
            result = sfpi::setexp(q, half) * sfpi::setexp(sfpi::vFloat(1.0f), exponent - half);

            // For exceptional inputs only the reciprocal's sign and zero/Inf
            // classification matter. Inverting its exponent gives Inf for zero,
            // zero for Inf/NaN, and a finite nonzero scale for every normal divisor.
            // Zero scales would hide NaN divisors, so add an all-ones NaN bit
            // pattern where |in1| > +Inf in sign-magnitude order, and +0 elsewhere.
            // Normal inputs have biased exponents in [1, 254], or unbiased exponents
            // in [-126, 127]; zero/subnormal and Inf/NaN fall outside. Each bound is a
            // sign test on SFPIADD, and the compares narrow lanes without SFPAND.
            v_if(!(ea >= 1 && ea < 255 && eb >= -126 && eb < 128)) {
                sfpi::vFloat scale = sfpi::setman(sfpi::as<sfpi::vFloat>(~sfpi::as<sfpi::vInt>(in1)), 0);
                sfpi::vFloat nan_divisor = sfpi::vFloat(__builtin_rvtt_sfpgt(
                    sfpi::setsgn(in1, 0).get(), sfpi::vFloat(std::numeric_limits<float>::infinity()).get(), 8));
                // Inverting in1 also inverts the scale's sign, so negate the product.
                result = nan_divisor - in0 * scale;
            }
            v_endif;
        } else {
            sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

            // sfpu_reciprocal_iter<2> written out, so that in0's load fills the bubble after the first multiply-add.
            sfpi::vFloat r = sfpi::approx_recip(in1);
            sfpi::vFloat t = in1 * r - sfpi::vConstFloatPrgm0;
            sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
            sfpi::vFloat y1 = r * -t - 0.0f;
            v_if(t < 0) {
                t = in1 * y1 - sfpi::vConstFloatPrgm0;
                r = y1 * -t - 0.0f;
            }
            v_endif;
            result = in0 * r;

            // Zero divisor: r = +inf, so a +0 dividend already gives in0 * r = NaN. vConstFloatPrgm1 = inf, from the init.
            v_if(in1 == 0 && in0 != 0) { result = sfpi::copysgn(sfpi::vFloat(sfpi::vConstFloatPrgm1), in0); }
            v_endif;

            // software RNE approach:
            result = float32_to_bf16_rne(result);
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE /*unused*/, BinaryOp BINOP>
inline void sfpu_binary_init() {
    if constexpr (BINOP == BinaryOp::DIV) {
        // Initialisation for sfpu_reciprocal_iter<2> in DIV and the zero-divisor infinity of the div arm.
        sfpu_reciprocal_init<false>();
        sfpi::vConstFloatPrgm1 = std::numeric_limits<float>::infinity();
    } else if constexpr (BINOP == BinaryOp::POW) {
        // Initialisation for use of sfpu_reciprocal_iter<2> in POW.
        sfpu_reciprocal_init<false>();
    } else if constexpr (BINOP == BinaryOp::XLOGY) {
        _init_log_<APPROXIMATION_MODE>();
    } else if constexpr (BINOP == BinaryOp::NEXTAFTER || BINOP == BinaryOp::NEXTAFTER_BF16) {
        sfpi::vConstIntPrgm0 = 0x7FFFFFFF;
        sfpi::vConstIntPrgm1 = 0x7F800000;
        sfpi::vConstIntPrgm2 = (BINOP == BinaryOp::NEXTAFTER_BF16) ? 0x10000 : 1;
    }
}

}  // namespace sfpu
}  // namespace ckernel
