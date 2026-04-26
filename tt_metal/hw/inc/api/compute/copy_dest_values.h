// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_copy_dest_values.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Copies all values from the tile in idst_in to the tile in idst_out in the DST register buffer.
 * This is a generalized version that takes a DataFormat template parameter.
 *
 * The DST register buffer must be in acquired state via *tile_regs_acquire* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                           | Type                     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|--------------------------|-------------------------------------------------------|----------|
 * | DATA_FORMAT    | The data format for copy operation                                    | DataFormat               | Any valid DataFormat enum value                       | True     |
 * | idst_in        | The index of the tile in DST register buffer to copy values from      | uint32_t                 | Must be less than the size of the DST register buffer | True     |
 * | idst_out       | The index of the tile in DST register buffer to copy values to        | uint32_t                 | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <DataFormat DATA_FORMAT>
ALWI void copy_dest_values(uint32_t idst_in, uint32_t idst_out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        copy_dest_value,
        (DATA_FORMAT, /*APPROXIMATE=*/false),
        idst_in,
        idst_out,
        0 /*unused*/,
        (int)VectorMode::RC)));
}

// clang-format off
/**
 * Copies all values from the tile in idst_in to the tile in idst_out in the DST register buffer.
 *
 * The DST register buffer must be in acquired state via *tile_regs_acquire* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst_in        | The index of the tile in DST register buffer to copy values from      | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst_out       | The index of the tile in DST register buffer to copy values to        | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
[[deprecated("Use copy_dest_values<DataFormat> instead")]]
ALWI void copy_dest_values(uint32_t idst_in, uint32_t idst_out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        copy_dest_value,
        (/*APPROXIMATE=*/false),
        idst_in,
        idst_out,
        0 /*unused*/,
        (int)VectorMode::RC)));
}

ALWI void copy_dest_values_init() { MATH((SFPU_BINARY_INIT_FN(unused, sfpu::copy_dest_value_init))); }

}  // namespace ckernel
