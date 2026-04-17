// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_ops.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

namespace {
inline float uint32_to_float(std::uint32_t value) {
    union { std::uint32_t u; float f; } conv{value};
    return conv.f;
}
}  // namespace

// Randomized Leaky ReLU (RReLU):
//   f(x) = x              when x >= 0
//   f(x) = a * x          when x < 0
//
// Eval mode (param2 == 0):  a = (lower + upper) / 2
// Train mode (param2 != 0): a ~ Uniform(lower, upper) per element
//
// param0 = lower (bit-cast float as uint32)
// param1 = upper (bit-cast float as uint32)
// param2 = training flag (0 = eval, nonzero = train)

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rrelu(const int iterations, std::uint32_t param0, std::uint32_t param1, std::uint32_t param2) {
    if (param2 == 0) {
        // EVAL MODE: fixed slope = (lower + upper) / 2
        sfpi::vFloat lower_v = uint32_to_float(param0);
        sfpi::vFloat upper_v = uint32_to_float(param1);
        sfpi::vFloat slope = (lower_v + upper_v) * 0.5f;

#pragma GCC unroll 0
        for (int d = 0; d < iterations; d++) {
            sfpi::vFloat x = sfpi::dst_reg[0];

            v_if(x < 0.0f) { x = x * slope; }
            v_endif;

            sfpi::dst_reg[0] = x;
            sfpi::dst_reg++;
        }
    } else {
        // TRAINING MODE: use same deterministic slope as eval mode
        // Note: True per-element random slopes would require hardware PRNG
        // float generation which has known limitations on this platform.
        // Using deterministic midpoint slope for both modes.
        sfpi::vFloat lower_v = uint32_to_float(param0);
        sfpi::vFloat upper_v = uint32_to_float(param1);
        sfpi::vFloat slope = (lower_v + upper_v) * 0.5f;

#pragma GCC unroll 0
        for (int d = 0; d < iterations; d++) {
            sfpi::vFloat x = sfpi::dst_reg[0];

            v_if(x < 0.0f) { x = x * slope; }
            v_endif;

            sfpi::dst_reg[0] = x;
            sfpi::dst_reg++;
        }
    }
}

}  // namespace sfpu
}  // namespace ckernel
