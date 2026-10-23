// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_relu.h"

namespace ckernel::sfpu {

// General template structure to implement activations
template <bool APPROXIMATION_MODE, ActivationType ACTIVATION_TYPE>
struct ActivationImpl;

// Specialization for HARDSIGMOID activation
template <bool APPROXIMATION_MODE>
struct ActivationImpl<APPROXIMATION_MODE, ActivationType::Hardsigmoid> {
    static inline void apply(sfpi::vFloat& v) {
        sfpi::vFloat tmp = (v * sfpi::vConstFloatPrgm0) + sfpi::vConstFloatPrgm1;
        v = _relu_max_body_(tmp, 1.0f);
    }
};

// Dispatch wrapper function
template <bool APPROXIMATION_MODE, ActivationType ACTIVATION_TYPE>
inline void apply_activation(sfpi::vFloat& v) {
    ActivationImpl<APPROXIMATION_MODE, ACTIVATION_TYPE>::apply(v);
}

bool bf16_dest_hardsigmoid();
template <int ITERATIONS>
void calculate_hardsigmoid_bf16();
void init_hardsigmoid_bf16();
// Whether BF16 DEST runs the generated hardsigmoid kernel as one call over the whole tile.
inline constexpr bool hardsigmoid_bf16_whole_tile = true;
// Sets up the generated BF16 hardsigmoid kernel for the instance it serves.
template <bool bf16_kernel>
inline void hardsigmoid_bf16_tile_init() {
    if constexpr (bf16_kernel) {
        init_hardsigmoid_bf16();
    }
}

template <bool APPROXIMATION_MODE, ActivationType ACTIVATION_TYPE, int ITERATIONS, bool is_fp32_dest_acc_en = true>
inline void calculate_activation() {
    if constexpr (!is_fp32_dest_acc_en && !APPROXIMATION_MODE && ITERATIONS == 32) {
        if (bf16_dest_hardsigmoid()) {
            calculate_hardsigmoid_bf16<ITERATIONS>();
            return;
        }
    }
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        apply_activation<APPROXIMATION_MODE, ACTIVATION_TYPE>(v);
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void hardsigmoid_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    // For hardsigmoid slope is 1/6, FP32 IEEE 754 representation.
    sfpi::vConstFloatPrgm0 = 0.1666666716337204f;
    sfpi::vConstFloatPrgm1 = 0.5f;
}

}  // namespace ckernel::sfpu

#include "ckernel_sfpu_hardsigmoid_bf16.h"
