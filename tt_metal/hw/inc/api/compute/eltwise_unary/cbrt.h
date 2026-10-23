// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_cbrt.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise cube root computation on input x , where x is each element of a tile
 * in DST register at index tile_index. The input can be of float data type. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only available on
 * the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void cbrt_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_cube_root,
        (APPROX, is_fp32_dest_acc_en, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void cbrt_tile_init() { MATH(SFPU_UNARY_INIT_FN(cbrt, sfpu::cube_root_init, (APPROX))); }

}  // namespace ckernel
