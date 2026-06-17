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
#include "ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_log.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Convert float32 to bfloat16 using IEEE 754 Round-to-Nearest-Even (RNE)
// This implements the "add 0x7fff + LSB" algorithm for correct tie-breaking
sfpi_inline sfpi::vFloat float32_to_bf16_rne(sfpi::vFloat in) {
    // Get the float32 bits as unsigned integer
    sfpi::vUInt bits = sfpi::reinterpret<sfpi::vUInt>(in);

    // Extract the LSB of what will become the bf16 mantissa (bit 16 of float32)
    // This is needed for the tie-breaker: round to even
    sfpi::vUInt lsb = (bits >> 16) & 1;

    // Add 0x7fff + lsb to implement RNE:
    // - If lower 16 bits > 0x8000: overflow → rounds up
    // - If lower 16 bits < 0x8000: no overflow → rounds down
    // - If lower 16 bits = 0x8000 (tie) and lsb=0: 0x7fff+0=0xffff, no overflow → stays even
    // - If lower 16 bits = 0x8000 (tie) and lsb=1: 0x7fff+1=0x8000, overflow → rounds up to even
    bits = bits + 0x7fffU + lsb;

    // Clear the lower 16 bits to get bf16 in upper 16 bits (bf16 format in float32)
    bits = bits & 0xFFFF0000U;

    // Reinterpret back as float
    return sfpi::reinterpret<sfpi::vFloat>(bits);
}

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
    sfpi::vFloat x = setexp(base, 127);  // set exp to exp bias (put base in range of 1-2)

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

    return result;
}

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_sfpu_binary(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
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

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_binary_mul(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat result = in0 * in1;

        if constexpr (!is_fp32_dest_acc_en) {
            // Software RNE approach (kept for reference):
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
inline void calculate_sfpu_binary_div(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat result = in0 * sfpu_reciprocal_iter<2>(in1);

        v_if(in1 == 0) {
            v_if(in0 == 0) { result = std::numeric_limits<float>::quiet_NaN(); }
            v_else {
                result = std::numeric_limits<float>::infinity();
                result = sfpi::copysgn(result, in0);
            }
            v_endif;
        }
        v_elseif(in0 == in1) { result = sfpi::vConst1; }
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
