// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"

// ckernel_sfpu_softcap.h and the SfpuType entry it needs exist only under
// hw/ckernels/blackhole, so the API is declared for Blackhole only. Including this header
// on another arch is harmless; calling softcap_tile there fails to compile.
#if defined(ARCH_BLACKHOLE)

#ifdef TRISC_MATH
#include "ckernel_sfpu_softcap.h"
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
 * The output is bounded by beta: near-linear for |x| << beta, saturating smoothly at
 * +/-beta rather than truncating. +/-Inf clamps to +/-beta; NaN does not propagate and
 * -0.0 returns +0.0, both as in tanh_tile.
 * The bound is exact in fp32; a bf16 pack rounds to nearest, so a beta that is not
 * bf16-representable can come back up to half a bf16 ULP above beta.
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void softcap_tile(uint32_t idst, uint32_t beta, uint32_t beta_recip) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        calculate_softcap,
        (APPROX, is_fp32_dest_acc_en, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC,
        beta,
        beta_recip));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void softcap_tile_init() {
    MATH(SFPU_UNARY_INIT(softcap, is_fp32_dest_acc_en));
}

}  // namespace ckernel

#endif  // ARCH_BLACKHOLE
