// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "ckernel_sfpu_selu.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise computation of selu = scale * (max(0,x) + min(0, alpha * (exp(x)-1))), where x is each
 * element of a tile in DST register at index tile_index. scale and alpha are each passed as the raw bits of a
 * float. The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is
 * only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | scale           | Scale used in selu calculation, as the raw bits of a float                 | uint32_t |                                                       | True     |
 * | alpha           | Alpha used in selu calculation, as the raw bits of a float                 | uint32_t |                                                       | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void selu_tile(uint32_t idst, uint32_t scale, uint32_t alpha) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_selu,
        (APPROX, is_fp32_dest_acc_en, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC,
        scale,
        alpha));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void selu_tile_pack(uint32_t idst, uint32_t scale, uint32_t alpha) {
    PACK(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_selu,
        (APPROX, is_fp32_dest_acc_en, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC,
        scale,
        alpha));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void selu_tile_init() { MATH(SFPU_UNARY_INIT(selu)); }

ALWI void selu_tile_init_pack() { PACK(SFPU_UNARY_INIT(selu)); }

}  // namespace ckernel
