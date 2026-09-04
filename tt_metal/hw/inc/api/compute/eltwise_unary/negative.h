/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_negative.h"
#endif

namespace ckernel {

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void negative_tile_init() {
    MATH((sfpu::Negative<APPROX, DataFormat::Float16_b, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}
// clang-format off
/**
 * Performs element-wise computation of the negative on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void negative_tile(uint32_t idst) {
    MATH((sfpu::Negative<APPROX, DataFormat::Float16_b, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC)));
}

#ifndef ARCH_QUASAR

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void negative_tile_int32(uint32_t idst) {
    MATH((sfpu::Negative<APPROX, DataFormat::Int32, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::RC)));
}

#endif
}  // namespace ckernel
