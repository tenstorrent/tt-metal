// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "llk_defs.h"

namespace ckernel::sfpu {

// General template structure to implement activations
template <bool APPROXIMATION_MODE, ActivationType ACTIVATION_TYPE>
struct ActivationImpl;

// Specialization for HARDSIGMOID activation
template <bool APPROXIMATION_MODE>
struct ActivationImpl<APPROXIMATION_MODE, ActivationType::Hardsigmoid> {
    static inline void apply(sfpi::vFloat& v) {
        sfpi::vFloat tmp = (v * sfpi::vConstFloatPrgm0) + sfpi::vConstFloatPrgm1;
        v_if(tmp > 1.0f) { tmp = 1.0f; }
        v_endif;
        v_if(tmp < 0.0f) { tmp = 0.0f; }
        v_endif;
        v = tmp;
    }
};

// Dispatch wrapper function
template <bool APPROXIMATION_MODE, ActivationType ACTIVATION_TYPE>
inline void apply_activation(sfpi::vFloat& v) {
    ActivationImpl<APPROXIMATION_MODE, ACTIVATION_TYPE>::apply(v);
}

template <bool APPROXIMATION_MODE, ActivationType ACTIVATION_TYPE, int ITERATIONS>
inline void calculate_activation() {
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
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
    // For hardsigmoid slope is 1/6, FP32 IEEE 754 representation.
    sfpi::vConstFloatPrgm0 = 0.1666666716337204f;
    sfpi::vConstFloatPrgm1 = 0.5f;
}

}  // namespace ckernel::sfpu
