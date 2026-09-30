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
            sfpi::vFloat tiny = sfpi::as<sfpi::vFloat>(sfpi::vInt(kUlpStep));
            v_if(mag_a == 0.0f && in1 > 0.0f) { result = tiny; }
            v_endif;
            v_if(mag_a == 0.0f && in1 < 0.0f) { result = -tiny; }
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
            constexpr int32_t kInfBits = 0x7F800000;
            constexpr int32_t kAbsMask = 0x7FFFFFFF;
            v_if((bits & kAbsMask) > kInfBits) { result = nan; }
            v_endif;
            v_if((sfpi::as<sfpi::vInt>(in1) & kAbsMask) > kInfBits) { result = nan; }
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
        // The pow arm keeps its RISC loop over the rows: unrolled, its long body idles 3 cycles per tile more, measured.
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
            // software RNE approach:
            result = float32_to_bf16_rne(result);
            // No zero guard: the SFPU multiply already returns +0 for a zero times any finite value (a denormal
            // counts as zero), so the only products a guard would change are 0 * inf and 0 * NaN, which are NaN.
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
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat r = sfpu_reciprocal_iter<2>(in1);
        sfpi::vFloat result = in0 * r;
        if constexpr (is_fp32_dest_acc_en) {
            // Skip quotient refinement when in0*r is already non-finite.
            // If in0*r = +/-inf, then the residual e = in0 - (+/-inf)*in1 = -/+inf and
            // result + e*r = inf + (-inf) = NaN, which would corrupt IEEE overflow behavior.
            v_if(sfpi::is_finite(result)) {
                // Residual (Markstein) refinement removes the double-rounding of in0 * round(1/in1).
                // The residual subtraction is exact under Sterbenz's lemma.
                sfpi::vFloat e = in0 - result * in1;
                result = result + e * r;
            }
            v_endif;
        }

        // Zero divisor: NaN for 0 / 0, a signed infinity otherwise. The two constants come from the programmable
        // constant registers that sfpu_binary_init<DIV> loads (vConstFloatPrgm1 = inf, vConstFloatPrgm2 = NaN), so
        // the arm loads no immediate per row; the values written are the same as before.
        v_if(in1 == 0) {
            result = sfpi::copysgn(sfpi::vFloat(sfpi::vConstFloatPrgm1), in0);
            v_if(in0 == 0) { result = sfpi::vConstFloatPrgm2; }
            v_endif;
        }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
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
        // Initialisation for use of sfpu_reciprocal_iter<2> in DIV (vConstFloatPrgm0 = 2.0), and the two answers of
        // the zero-divisor arm of calculate_sfpu_binary_div.
        sfpu_reciprocal_init<false>();
        sfpi::vConstFloatPrgm1 = std::numeric_limits<float>::infinity();
        sfpi::vConstFloatPrgm2 = std::numeric_limits<float>::quiet_NaN();
    } else if constexpr (BINOP == BinaryOp::POW) {
        // Initialisation for use of sfpu_reciprocal_iter<2> in POW.
        sfpu_reciprocal_init<false>();
    } else if constexpr (BINOP == BinaryOp::XLOGY) {
        _init_log_<APPROXIMATION_MODE>();
    }
}

}  // namespace sfpu
}  // namespace ckernel
