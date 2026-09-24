// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#ifndef ARCH_QUASAR
#include "ckernel_sfpu_elu.h"
#endif
#endif

namespace ckernel {
// Quasar has no elu kernel.
#ifndef ARCH_QUASAR
// clang-format off
/**
 * Performs element-wise computation of elu (relu(x) + slope*(exp(x) - 1)*(x <= 0 )) on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | tile_index     | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | slope          | slope used in elu calculation                                              | uint32_t | Greater than 0                                        | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void elu_tile(std::uint32_t idst, std::uint32_t param0) {
    MATH((sfpu::Elu<APPROX, is_fp32_dest_acc_en>::run(idst, param0)));
}
/**
 * Please refer to documentation for any_init.
 */
ALWI void elu_tile_init() { MATH((sfpu::Elu<APPROX, DST_ACCUM_MODE>::init())); }
#endif
}  // namespace ckernel
