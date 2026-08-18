// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// hardsigmoid(x) = clamp(x/6 + 1/2, 0, 1)
constexpr float HARDSIGMOID_SLOPE = 0.1666666716337204f;  // 1/6 in fp32 (0x3E2AAAAB); not bf16-exact
constexpr float HARDSIGMOID_SHIFT = 0.5f;
constexpr float HARDSIGMOID_UPPER = 1.0f;  // clamp ceiling, reached at x = 3
constexpr float HARDSIGMOID_LOWER = 0.0f;  // clamp floor, reached at x = -3

// General template structure to implement activations. The primary template is left undefined, so an
// activation without a specialization is a compile error.
template <bool APPROXIMATION_MODE, ActivationType ACTIVATION_TYPE>
struct ActivationImpl;

// Two-sided clamp to [0, 1]. Quasar has no equivalent of Blackhole's _relu_max_body_ (its relu family
// is raw TTI), and sfpi::min / sfpi::max lower to SFPSWAP without the float-compare bit. Compares are
// strict because a float >= is unreliable at the boundary itself on Quasar (issue #50208); both knees
// land exactly on 1.0 / 0.0, so > / < are exact there.
sfpi_inline sfpi::vFloat hardsigmoid_clamp(sfpi::vFloat val) {
    sfpi::vFloat result = val;
    v_if(result > HARDSIGMOID_UPPER) { result = HARDSIGMOID_UPPER; }
    v_endif;
    v_if(result < HARDSIGMOID_LOWER) { result = HARDSIGMOID_LOWER; }
    v_endif;
    return result;
}

// Specialization for the HARDSIGMOID activation
template <bool APPROXIMATION_MODE>
struct ActivationImpl<APPROXIMATION_MODE, ActivationType::Hardsigmoid> {
    static inline void apply(sfpi::vFloat& v) {
        // affine ramp x/6 + 1/2 in one SFPMAD over the constants hardsigmoid_init programmed
        sfpi::vFloat tmp = (v * sfpi::vConstFloatPrgm0) + sfpi::vConstFloatPrgm1;
        v = hardsigmoid_clamp(tmp);
    }
};

// Dispatch wrapper function
template <bool APPROXIMATION_MODE, ActivationType ACTIVATION_TYPE>
inline void apply_activation(sfpi::vFloat& v) {
    ActivationImpl<APPROXIMATION_MODE, ACTIVATION_TYPE>::apply(v);
}

/**
 * @brief Program the fp32 ramp constants the Hardsigmoid body reads.
 *
 * @tparam APPROXIMATION_MODE: accepted for ABI parity but ignored (hardsigmoid is exact).
 * @note The surrounding @ref llk_math_eltwise_unary_sfpu_init resets the counters, so unlike the
 *       Blackhole kernel this init only programs constants.
 */
template <bool APPROXIMATION_MODE>
inline void hardsigmoid_init() {
    sfpi::vConstFloatPrgm0 = HARDSIGMOID_SLOPE;
    sfpi::vConstFloatPrgm1 = HARDSIGMOID_SHIFT;
}

/**
 * @brief Apply an activation in-place over a Dest tile.
 *
 * @tparam APPROXIMATION_MODE: select the approximate body where an activation offers one; Hardsigmoid
 *         ignores it (it is exact).
 * @tparam ACTIVATION_TYPE: activation to apply; picks the @ref ActivationImpl body at compile time.
 * @tparam ITERATIONS: Number of SFPU loop iterations over the Dest tile.
 * @note For ActivationType::Hardsigmoid, call @ref hardsigmoid_init first — it programs the ramp
 *       constants (vConstFloatPrgm0/1) the body reads.
 */
template <bool APPROXIMATION_MODE, ActivationType ACTIVATION_TYPE, int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_activation() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        apply_activation<APPROXIMATION_MODE, ACTIVATION_TYPE>(v);
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
