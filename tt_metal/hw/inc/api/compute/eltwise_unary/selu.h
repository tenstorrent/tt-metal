// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#ifndef ARCH_QUASAR
#include "ckernel_sfpu_selu.h"
#endif
#endif

namespace ckernel {

// Quasar has no selu kernel.
#ifndef ARCH_QUASAR
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
ALWI void selu_tile(std::uint32_t idst, std::uint32_t scale, std::uint32_t alpha) {
    MATH((sfpu::Selu<APPROX, is_fp32_dest_acc_en>::run(idst, scale, alpha)));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void selu_tile_pack(std::uint32_t idst, std::uint32_t scale, std::uint32_t alpha) {
    PACK((sfpu::Selu<APPROX, is_fp32_dest_acc_en>::run(idst, scale, alpha)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void selu_tile_init() { MATH((sfpu::Selu<APPROX, DST_ACCUM_MODE>::init())); }

ALWI void selu_tile_init_pack() { PACK((sfpu::Selu<APPROX, DST_ACCUM_MODE>::init())); }
#endif  // !ARCH_QUASAR

}  // namespace ckernel
