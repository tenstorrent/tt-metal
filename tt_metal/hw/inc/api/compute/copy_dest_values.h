// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
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
template <DataFormat DATA_FORMAT, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void copy_dest_values(uint32_t idst_in, uint32_t idst_out) {
    MATH((sfpu::CopyDestValues<DATA_FORMAT, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst_in, idst_out, 0 /*unused*/, VectorMode::RC)));
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
[[deprecated("Use copy_dest_values<DataFormat> instead")]]
ALWI void copy_dest_values(uint32_t idst_in, uint32_t idst_out) {
    // Routes through the deprecated 1-template-arg `copy_dest_value<APPROXIMATE>` overload in
    // ckernel::sfpu (the format-agnostic sfpi::vFloat path), selected by DataFormat::Invalid.
    // New code should use the DataFormat-templated overload above.
    MATH((sfpu::CopyDestValues<DataFormat::Invalid, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst_in, idst_out, 0 /*unused*/, VectorMode::RC)));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void copy_dest_values_init() {
    MATH((sfpu::CopyDestValues<DataFormat::Float16_b, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

}  // namespace ckernel
