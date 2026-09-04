// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_identity.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs a simple elementwise copy / identity operation on the input: y(x) = x
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform identity operation | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void identity_tile(uint32_t idst) {
    MATH((sfpu::Identity<APPROX, DataFormat::Float16_b, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void identity_tile_init() {
    MATH((sfpu::Identity<APPROX, DataFormat::Float16_b, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

// clang-format off
/**
 * Performs a simple elementwise copy / identity operation on the input: y(x) = x
 * This function should be used with unsigned integer formats: uint32
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform identity operation | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void identity_tile_uint32(uint32_t idst) {
    MATH((sfpu::Identity<APPROX, DataFormat::UInt32, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC)));
}

}  // namespace ckernel
