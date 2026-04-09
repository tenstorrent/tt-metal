// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// RReLU evaluation mode: x >= 0 ? x : slope * x
// slope is pre-computed by host as (lower + upper) / 2 and passed in bfloat16 format.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rrelu_eval(uint32_t slope_bf16) {
    sfpi::vFloat slope = sfpi::s2vFloat16b(slope_bf16);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat result = x;

        v_if(x < 0.0f) { result = x * slope; }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// RReLU training mode: x >= 0 ? x : a * x
// where a ~ Uniform(lower, upper) per element using hardware PRNG.
// lower_bf16 and upper_bf16 are in bfloat16 format.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rrelu_train(uint32_t lower_bf16, uint32_t upper_bf16) {
    sfpi::vFloat lower_val = sfpi::s2vFloat16b(lower_bf16);
    sfpi::vFloat range = sfpi::s2vFloat16b(upper_bf16) - lower_val;

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // Generate random uint32 per SFPU lane via hardware PRNG.
        // SFPMOV with instr_mod1=8, lreg_c=9 generates pseudo-random values.
        TTI_SFPMOV(0, 9, p_sfpu::LREG3, 8);

        // Read random bits from LREG3 as unsigned integer
        sfpi::vUInt rand_bits(sfpi::l_reg[sfpi::LRegs::LReg3]);

        // Create uniform float in [1.0, 2.0) by:
        // - Keeping lower 23 bits as mantissa
        // - Setting exponent = 127 (IEEE 754 bias for 2^0 = 1.0)
        sfpi::vUInt float_bits = (rand_bits & 0x007FFFFFu) | 0x3F800000u;
        sfpi::vFloat uniform01 = sfpi::reinterpret<sfpi::vFloat>(float_bits) - 1.0f;

        // Scale to [lower, upper]: a = lower + uniform01 * range
        sfpi::vFloat a = lower_val + uniform01 * range;

        // Apply: x >= 0 ? x : a * x
        sfpi::vFloat result = x;
        v_if(x < 0.0f) { result = x * a; }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
