// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_rsqrt.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
template <bool legacy_compat = false>
ALWI void rsqrt_tile_init() {
#if defined(ARCH_QUASAR)
    static_assert(!legacy_compat, "legacy_compat is not supported in Quasar rsqrt");
    // Quasar computes rsqrt via the SFPU nonlinear unit (SQRT then RECIP); no LUT init is required.
    MATH(SFPU_UNARY_INIT(rsqrt));
#else
    MATH(SFPU_UNARY_INIT_FN(rsqrt, sfpu::rsqrt_init, (APPROX, legacy_compat)));
#endif
}

// clang-format off
/**
 * Performs element-wise computation of reciprocal sqrt on each element of a
 * tile in DST register at index idst. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool legacy_compat = false, bool FAST_APPROX = false>
ALWI void rsqrt_tile(uint32_t idst) {
#if defined(ARCH_QUASAR)
    static_assert(!legacy_compat, "legacy_compat is not supported in Quasar rsqrt");
    static_assert(!FAST_APPROX, "FAST_APPROX is not supported in Quasar rsqrt");
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, _calculate_rsqrt_, (8 /* ITERATIONS */), idst, VectorMode::RC));
#else
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_rsqrt,
        (APPROX, 8 /* ITERATIONS */, DST_ACCUM_MODE, FAST_APPROX, legacy_compat),
        idst,
        VectorMode::RC));
#endif
}

}  // namespace ckernel
