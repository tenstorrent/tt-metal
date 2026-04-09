// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

using namespace sfpi;

namespace ckernel::sfpu {

// Namespace-scoped PRNG state for training mode.
// Advances each iteration to produce different random slopes across tiles.
static uint32_t rrelu_rng_state = 0xDEADBEEF;

// Randomized Leaky ReLU (RReLU):
//   f(x) = x              when x >= 0
//   f(x) = a * x          when x < 0
//
// Eval mode  (training == 0): a = (lower + upper) / 2    (stored in vConstFloatPrgm2)
// Train mode (training != 0): a ~ Uniform(lower, upper)  per element
//
// lower is in vConstFloatPrgm0, upper is in vConstFloatPrgm1,
// midpoint = (lower + upper) / 2 is in vConstFloatPrgm2.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rrelu(uint training) {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[0];
        vFloat result = x;

        if (training == 0) {
            // Eval mode: fixed slope = midpoint = (lower + upper) / 2
            v_if(x < 0.0f) { result = x * vConstFloatPrgm2; }
            v_endif;
        } else {
            // Training mode: per-element random slope in [lower, upper]
            // Advance scalar RNG state (LCG step)
            rrelu_rng_state ^= rrelu_rng_state << 13;
            rrelu_rng_state ^= rrelu_rng_state >> 17;
            rrelu_rng_state ^= rrelu_rng_state << 5;

            // Mix input bits with global state for per-lane diversity
            vInt bits = reinterpret<vInt>(x);
            bits = bits ^ vInt(rrelu_rng_state);
            // Additional mixing passes
            bits = bits ^ (bits >> 16);
            bits = bits ^ (bits << 7);
            bits = bits ^ (bits >> 13);

            // Convert to float in [0, 1):
            // Take low 23 bits as mantissa, set exponent to 127 -> [1.0, 2.0), subtract 1.0
            vInt mantissa = bits & vInt(0x007FFFFF);
            vFloat rand_val = reinterpret<vFloat>(mantissa | vInt(0x3F800000));
            rand_val = rand_val - vConst1;

            // Scale to [lower, upper]: slope = rand_val * (upper - lower) + lower
            vFloat range = vConstFloatPrgm1 - vConstFloatPrgm0;
            vFloat slope = rand_val * range + vConstFloatPrgm0;

            v_if(x < 0.0f) { result = x * slope; }
            v_endif;
        }

        dst_reg[0] = result;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void rrelu_init(uint lower_bits, uint upper_bits) {
    // Bit-cast uint32_t parameters to float (aliasing-safe)
    float lower, upper;
    __builtin_memcpy(&lower, &lower_bits, sizeof(float));
    __builtin_memcpy(&upper, &upper_bits, sizeof(float));
    vConstFloatPrgm0 = lower;
    vConstFloatPrgm1 = upper;
    vConstFloatPrgm2 = (lower + upper) * 0.5f;
}

}  // namespace ckernel::sfpu
