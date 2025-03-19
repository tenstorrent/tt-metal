/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_mask.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

ALWI void mask_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_mask_init<true>()));  // TODO(AP): move out init
}

// clang-format off
/**
 * Performs element-wise computation of mask on each element of a tile
 * in data and mask DST register. *mask_tile* will mask each element with 0,
 * *mask_posinf_tile* will mask each element with *float(inf)*.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 *
 * TODO: fix idst2_mask.
 * currently idst2_mask is not used and (idst_data + 1) is used for mask.
 * because don't know how to use 2 dst register with sfpu.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type       | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|------------|-------------------------------------------------------|----------|
 * | dst_data_index | The index of the tile in DST REG for the data and result                   | uint32_t   | Must be less than the acquired size of DST REG        | True     | 
 * | dst_mask_index | The index of the tile in DST REG for the mask                              | uint32_t   | Must be less than the acquired size of DST REG        | True     |
 * | data_format    | The format of the data and mask (supports Float16, Float16_b, and Int32)   | DataFormat | Must be a valid data format                           | False    |
 */
 // clang-format on
ALWI void mask_tile(uint32_t idst_data, uint32_t idst2_mask, DataFormat data_format = DataFormat::Float16_b) {
    MATH((llk_math_eltwise_unary_sfpu_mask<true>(idst_data, data_format)));
}

ALWI void mask_posinf_tile(uint32_t idst_data, uint32_t idst2_mask) {
    MATH((llk_math_eltwise_unary_sfpu_mask_posinf<true>(idst_data)));
}

}  // namespace ckernel
