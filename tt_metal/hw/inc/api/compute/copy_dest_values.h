// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_copy_dest_values.h"
#include "ckernel_sfpu_copy_dest_values.h"
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
    MATH(llk_math_eltwise_binary_sfpu_copy_dest_values<static_cast<DataFormat>(DATA_FORMAT)>(idst_in, idst_out));
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
    MATH(llk_math_eltwise_binary_sfpu_copy_dest_values(idst_in, idst_out));
}

ALWI void copy_dest_values_init() { MATH(llk_math_eltwise_binary_sfpu_copy_dest_values_init()); }

}  // namespace ckernel
