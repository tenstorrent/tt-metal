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
        // Clamp before scaling: (v + 3) is exact for v near -3 by Sterbenz's lemma, so
        // multiplying the clamped value by 1/6 avoids the catastrophic cancellation that
        // v * (1/6) + 0.5 suffers there (up to 25% relative error just below v = -3).
        // Matches torch's evaluation order: clamp(v + 3, 0, 6) / 6.
        sfpi::vFloat tmp = _relu_max_body_(v + 3.0f, 6.0f);
        v = tmp * sfpi::vConstFloatPrgm0;
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
    math::reset_counters(p_setrwc::SET_ABD_F);
    // For hardsigmoid slope is 1/6, FP32 IEEE 754 representation.
    sfpi::vConstFloatPrgm0 = 0.1666666716337204f;
}

}  // namespace ckernel::sfpu
