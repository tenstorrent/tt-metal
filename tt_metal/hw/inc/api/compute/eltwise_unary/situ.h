// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_situ.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise computation of softcap(x) = beta * tanh(x / beta) on
 * each element of a tile in DST register at index tile_index. The DST register
 * buffer must be in acquired state via *acquire_dst* call. This call is blocking
 * and is only available on the compute engine.
 *
 * This is the up half of Moonshot's SiTU activation. tanh is bounded by 1, so the
 * output is bounded by beta: the op saturates smoothly rather than truncating, staying
 * near-linear for |x| << beta and saturating at +/-beta. +/-Inf clamps to
 * +/-beta; NaN does not propagate and yields a finite result.
 *
 * beta is passed as an fp32 bit pattern, with its reciprocal precomputed by the
 * caller so the kernel performs no division.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | beta           | The beta parameter, as an fp32 bit pattern                                 | uint32_t | Must be non-zero                                      | True     |
 * | beta_recip     | The reciprocal of beta, as an fp32 bit pattern                             | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void softcap_tile(uint32_t idst, uint32_t beta, uint32_t beta_recip) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_softcap,
        (APPROX, DST_ACCUM_MODE, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC,
        beta,
        beta_recip));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void softcap_tile_init() { MATH(SFPU_UNARY_INIT(softcap)); }

// clang-format off
/**
 * Performs element-wise computation of situ_gate(x) = beta * tanh(x / beta) * sigmoid(x)
 * on each element of a tile in DST register at index tile_index. The DST register
 * buffer must be in acquired state via *acquire_dst* call. This call is blocking
 * and is only available on the compute engine.
 *
 * This is the gate half of Moonshot's SiTU activation. sigmoid takes the raw x,
 * not the capped value, so the output is bounded by beta and is suppressed
 * toward zero for negative x. +/-Inf yields +beta / -0; NaN does not propagate
 * and yields a finite result.
 *
 * The full activation is situ_gate(gate) * softcap(up), the product being an
 * ordinary elementwise multiply outside these ops.
 *
 * beta is passed as an fp32 bit pattern, with its reciprocal precomputed by the
 * caller so the kernel performs no division.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | beta           | The beta parameter, as an fp32 bit pattern                                 | uint32_t | Must be non-zero                                      | True     |
 * | beta_recip     | The reciprocal of beta, as an fp32 bit pattern                             | uint32_t |                                                       | True     |
 */
// clang-format on
ALWI void situ_gate_tile(uint32_t idst, uint32_t beta, uint32_t beta_recip) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_situ_gate,
        (APPROX, DST_ACCUM_MODE, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC,
        beta,
        beta_recip));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void situ_gate_tile_init() { MATH(SFPU_UNARY_INIT(situ_gate)); }

}  // namespace ckernel
