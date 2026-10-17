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
#include "ckernel_sfpu_binary_pow.h"
#include "sfpu/ckernel_sfpu_log.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

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
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
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
            // The kernel ttnn.pow runs (calculate_sfpu_binary_pow). This op used to carry its
            // own copy built on the fp16-rounded ln(2) = 0.692871 and the quadratic _sfpu_exp_
            // with repeated squaring, which was 4-28% off for |log2(base)| >= 10 on Blackhole.
            result = _sfpu_binary_power_<is_fp32_dest_acc_en>(in0, in1);
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
    }
}

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_binary_mul(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr std::uint32_t dst_tile_size_sfpi = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat result = in0 * in1;

        if constexpr (!is_fp32_dest_acc_en) {
            // software RNE approach:
            result = float32_to_bf16_rne(result);

            // To match FPU behaviour for bfloat16 multiplication, 0 * x = 0 and x * 0 = 0
            v_if(in0 == 0 || in1 == 0) { result = 0.0f; }
            v_endif;
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
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat result;
        if constexpr (is_fp32_dest_acc_en) {
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
            result = in0 * sfpu_reciprocal_iter<2>(in1);
            v_if(in1 == 0) {
                v_if(in0 == 0) { result = std::numeric_limits<float>::quiet_NaN(); }
                v_else {
                    result = std::numeric_limits<float>::infinity();
                    result = sfpi::copysgn(result, in0);
                }
                v_endif;
            }
            v_endif;
        }

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
        // Initialisation for use of sfpu_reciprocal_iter<2> in DIV.
        sfpu_reciprocal_init<false>();
    } else if constexpr (BINOP == BinaryOp::POW) {
        sfpu_binary_pow_init<APPROXIMATION_MODE>();
    } else if constexpr (BINOP == BinaryOp::XLOGY) {
        _init_log_<APPROXIMATION_MODE>();
    }
}

}  // namespace sfpu
}  // namespace ckernel
