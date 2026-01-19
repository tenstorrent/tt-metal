// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

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

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_sfpu_binary(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    _calculate_sfpu_binary_<APPROXIMATION_MODE, BINOP, ITERATIONS>(dst_index_in0, dst_index_in1, dst_index_out);
}

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_binary_mul(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    for (int d = 0; d < ITERATIONS; d++) {
        // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
        constexpr uint dst_tile_size_sfpi = 32;
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat result = in0 * in1;

        if constexpr (!is_fp32_dest_acc_en) {
            // // Pre-subtract tie-breaker to compensate for float_to_fp16b using 0x8000 instead of 0x7fff
            // // Skip for zero to avoid underflow (0x00000000 - 1 = garbage)
            // sfpi::vUInt bits = sfpi::reinterpret<sfpi::vUInt>(result);
            // sfpi::vUInt lsb = ((~bits) >> 16) & 1;
            // bits = bits - lsb;

            // result = sfpi::reinterpret<sfpi::vFloat>(bits);
            // result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));

            // sfpi::vUInt in0_bits = sfpi::reinterpret<sfpi::vUInt>(in0);
            // sfpi::vUInt in1_bits = sfpi::reinterpret<sfpi::vUInt>(in1);
            // sfpi::vInt in0_is_zero = (in0_bits & 0x7FFFFFFFU);  // catches +0.0 AND -0.0
            // sfpi::vInt in1_is_zero = (in1_bits & 0x7FFFFFFFU);
            // v_if( (in0_is_zero   == 0) || (in1_is_zero == 0) ) { result = 0.0f; }
            // v_endif;

            // Old software RNE approach (kept for reference):
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
    for (int d = 0; d < ITERATIONS; d++) {
        // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
        constexpr uint dst_tile_size_sfpi = 32;
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat result = 0.0f;

        v_if(in1 == 0) {
            v_if(in0 == 0) { result = std::numeric_limits<float>::quiet_NaN(); }
            v_else {
                result = std::numeric_limits<float>::infinity();
                result = sfpi::setsgn(result, in0);
            }
            v_endif;
        }
        v_elseif(in0 == in1) { result = sfpi::vConst1; }
        v_else { result = in0 * _sfpu_reciprocal_<2>(in1); }
        v_endif;

        // Apply RNE rounding outside conditional block to avoid compiler ICE
        if constexpr (!is_fp32_dest_acc_en) {
            // // Pre-subtract tie-breaker to compensate for float_to_fp16b using 0x8000 instead of 0x7fff
            // // Skip for zero to avoid underflow (0x00000000 - 1 = garbage)
            // sfpi::vUInt bits = sfpi::reinterpret<sfpi::vUInt>(result);
            // sfpi::vUInt lsb = ((~bits) >> 16) & 1;
            // bits = bits - lsb;

            // result = sfpi::reinterpret<sfpi::vFloat>(bits);
            // result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));

            // sfpi::vUInt in0_bits = sfpi::reinterpret<sfpi::vUInt>(in0);
            // sfpi::vUInt in1_bits = sfpi::reinterpret<sfpi::vUInt>(in1);
            // sfpi::vInt in0_is_zero = (in0_bits & 0x7FFFFFFFU);  // catches +0.0 AND -0.0
            // sfpi::vInt in1_is_zero = (in1_bits & 0x7FFFFFFFU);
            // v_if( (in0_is_zero   == 0) || (in1_is_zero == 0) ) { result = 0.0f; }
            // v_endif;

            // software RNE approach (kept for reference):
            result = float32_to_bf16_rne(result);

            // // Restore zero if original was zero (avoid underflow corruption)
            // v_if(original_result == 0.0f) { result = 0.0f; }
            // v_endif;
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE /*unused*/, BinaryOp BINOP>
inline void sfpu_binary_init() {
    _sfpu_binary_init_<APPROXIMATION_MODE, BINOP>();
}

}  // namespace sfpu
}  // namespace ckernel
