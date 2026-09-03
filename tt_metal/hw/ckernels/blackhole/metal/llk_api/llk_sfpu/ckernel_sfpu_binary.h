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

    // 3rd order polynomial approx - determined using rminimax over [1,2]
    sfpi::vFloat series_result = x * (x * (x * 0x2.44734p-4f - 0xd.e712ap-4f) + 0x2.4f5388p+0f) - 0x1.952992p+0f;

    // Convert exponent to float
    sfpi::vSMag exp = sfpi::convert<sfpi::vSMag>(exexp(base));
    sfpi::vFloat expf = sfpi::convert<sfpi::vFloat>(exp, sfpi::RoundMode::Nearest);

    // De-normalize to original range
    sfpi::vFloat vConstLn2 = 0.692871f;
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
            result = calculate_sfpu_binary_power(in0, in1);
        } else if constexpr (BINOP == BinaryOp::XLOGY) {
            v_if((in1 < 0.0f) || (in1 == nan)) { result = nan; }
            v_else {
                sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = in1;
                _calculate_log_body_<false>(0, dst_index_out);
                result = sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] * in0;
            }
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

        sfpi::vFloat r = sfpu_reciprocal_iter<2>(in1);
        sfpi::vFloat result = in0 * r;
        if constexpr (is_fp32_dest_acc_en) {
            // Skip quotient refinement when in0*r is already non-finite (biased exponent == 255).
            // If in0*r = +/-inf, then the residual e = in0 - (+/-inf)*in1 = -/+inf and
            // result + e*r = inf + (-inf) = NaN, which would corrupt IEEE overflow behavior.
            v_if(sfpi::exexp(result, sfpi::ExponentMode::Biased) != 255) {
                // The residual is equally unusable when in1 is non-finite, and that case
                // reaches here because the quotient is finite rather than in spite of it:
                // r = 1/inf = 0 and result = in0 * 0 = 0, so result * in1 is 0 * inf, the
                // residual is NaN and the refinement destroys a correct zero. The guard is
                // about whether the residual can be formed, not about the quotient's size.
                v_and(sfpi::exexp(in1, sfpi::ExponentMode::Biased) != 255);
                // Residual (Markstein) refinement removes the double-rounding of in0 * round(1/in1).
                // The residual subtraction is exact under Sterbenz's lemma.
                sfpi::vFloat e = in0 - result * in1;
                result = result + e * r;
            }
            v_endif;
        }

        // The zero and non-finite arms below test magnitudes rather than the values:
        // the SFPU compare does not read -0.0 as equal to 0.0, so in0 == 0 misses a
        // negative zero dividend and -0.0 / 0.0 came back -inf instead of NaN.
        sfpi::vFloat abs_in0 = sfpi::setsgn(in0, 0);
        sfpi::vFloat abs_in1 = sfpi::setsgn(in1, 0);
        sfpi::vFloat vinf = std::numeric_limits<float>::infinity();

        v_if(abs_in1 == 0.0f) {
            v_if(abs_in0 == 0.0f) { result = std::numeric_limits<float>::quiet_NaN(); }
            v_else {
                result = vinf;
                result = sfpi::copysgn(result, in0);
                result = sfpi::copysgn(result, sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(in0) ^ sfpi::as<sfpi::vInt>(in1)));
            }
            v_endif;
        }
        v_endif;

        // A finite dividend over an infinite divisor is a signed zero, and the sign is
        // the exclusive or of the operand signs. The multiply that produced it does not
        // carry that sign, so put it back here.
        v_if(sfpi::as<sfpi::vInt>(abs_in1) == sfpi::as<sfpi::vInt>(vinf)) {
            v_if(sfpi::as<sfpi::vInt>(abs_in0) < sfpi::as<sfpi::vInt>(vinf)) {
                result = 0.0f;
                result = sfpi::copysgn(result, sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(in0) ^ sfpi::as<sfpi::vInt>(in1)));
            }
            v_endif;
        }
        v_endif;

        // NaN in either operand propagates. It used to come out of the residual step by
        // accident, so skipping that step for a non-finite divisor lost it for 0 / NaN.
        v_if(sfpi::as<sfpi::vInt>(abs_in0) > sfpi::as<sfpi::vInt>(vinf)) { result = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
        v_if(sfpi::as<sfpi::vInt>(abs_in1) > sfpi::as<sfpi::vInt>(vinf)) { result = std::numeric_limits<float>::quiet_NaN(); }
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
    if constexpr (BINOP == BinaryOp::DIV || BINOP == BinaryOp::POW) {
        // Initialisation for use of sfpu_reciprocal_iter<2> in DIV or POW.
        sfpu_reciprocal_init<false>();
    } else if constexpr (BINOP == BinaryOp::XLOGY) {
        _init_log_<APPROXIMATION_MODE>();
    }
}

}  // namespace sfpu
}  // namespace ckernel
