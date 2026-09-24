// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_xielu.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise xIELU computation on input x , where x is each element of a tile
 * in DST register at index tile_index. The input can be of float data type. The
 * DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on
 * the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                 | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to perform xIELU operation     | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | alpha_p        | alpha positive parameter                                                    | uint32_t |                                                       | True     |
 * | alpha_n        | alpha negative parameter                                                    | uint32_t |                                                       | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void xielu_tile(std::uint32_t idst, std::uint32_t alpha_p, std::uint32_t alpha_n) {
    MATH((sfpu::Xielu<APPROX, is_fp32_dest_acc_en>::run(idst, alpha_p, alpha_n)));
}

ALWI void xielu_tile_init() { MATH((sfpu::Xielu<APPROX, DST_ACCUM_MODE>::init())); }

}  // namespace ckernel
