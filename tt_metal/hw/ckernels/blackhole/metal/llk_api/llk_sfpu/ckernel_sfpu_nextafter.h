// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_binary_comp.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Element-wise nextafter(a, b): returns the next representable float after `a`
// in the direction of `b`, matching torch.nextafter / IEEE-754. The core step
// is +/- 1 ULP on the integer representation of `a`, with sign-aware reversal
// for negative inputs and explicit handling for ±0, ±inf, and NaN inputs.
//
// DATA_FORMAT picks the ULP magnitude: float32 uses 1, bfloat16 (Float16_b)
// uses 0x10000 because the bf16 value occupies the high 16 bits of the fp32
// representation in DST.
template <bool APPROXIMATION_MODE, int ITERATIONS, DataFormat DATA_FORMAT = DataFormat::Float32>
inline void calculate_sfpu_nextafter(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr uint dst_tile_size_sfpi = 32;
    constexpr int ULP_STEP_VAL = (DATA_FORMAT == DataFormat::Float16_b) ? 0x00010000 : 0x00000001;

    // sfpi lowers `vInt + <large_int_literal>` through a 12-bit signed-immediate
    // add and silently mis-encodes constants that don't fit (the bf16 step
    // 0x00010000 ends up flipping the sign bit). Materialising the step into a
    // vInt forces a register-register add. Bitwise &, |, == with int literals
    // compile correctly, so other constants stay inline.
    const sfpi::vInt ulp_step(ULP_STEP_VAL);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat b = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vInt a_bits = sfpi::reinterpret<sfpi::vInt>(a);
        sfpi::vInt b_bits = sfpi::reinterpret<sfpi::vInt>(b);
        sfpi::vInt a_abs_bits = a_bits & 0x7FFFFFFF;
        sfpi::vInt b_abs_bits = b_bits & 0x7FFFFFFF;

        // Default covers a == b (incl. ±0 == ±0) and any lane the special
        // cases below don't touch.
        sfpi::vInt result = b_bits;

        // Positive nonzero `a`: +ULP toward larger b, -ULP toward smaller b.
        v_if(a > 0.0f) {
            v_if(a < b) { result = a_bits + ulp_step; }
            v_elseif(a > b) { result = a_bits - ulp_step; }
            v_endif;
        }
        v_endif;

        // Negative nonzero `a`: integer ordering of negative floats is
        // reversed, so directions flip vs. the positive case.
        v_if(a < 0.0f) {
            v_if(a < b) { result = a_bits - ulp_step; }
            v_elseif(a > b) { result = a_bits + ulp_step; }
            v_endif;
        }
        v_endif;

        // |a| == 0: bit-step can't cross the sign boundary, so emit smallest
        // subnormal with sign(b). a == 0 == b leaves the default (b_bits)
        // intact so signed-zero semantics match `b`.
        v_if(a_abs_bits == 0) {
            v_if(b_abs_bits != 0) {
                sfpi::vInt b_sign = b_bits & 0x80000000;
                result = b_sign | ulp_step;
            }
            v_endif;
        }
        v_endif;

        // SFPU float comparisons are unreliable for ±inf, so handle every
        // inf-a lane with pure bit arithmetic:
        //   a == b           -> b_bits           (no step)
        //   a is inf, a != b -> a_bits - 1 ULP   (works for both ±inf endpoints)
        v_if(is_inf(a_abs_bits)) {
            result = b_bits;
            v_if(a_bits != b_bits) { result = a_bits - ulp_step; }
            v_endif;
        }
        v_endif;

        // NaN propagation: preserve the input NaN bit pattern; `a` wins when
        // both operands are NaN.
        v_if(is_nan(b_abs_bits)) { result = b_bits; }
        v_endif;
        v_if(is_nan(a_abs_bits)) { result = a_bits; }
        v_endif;

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = sfpi::reinterpret<sfpi::vFloat>(result);
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
