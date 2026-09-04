// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "ckernel_sfpu_hardtanh.h"
#endif

namespace ckernel {

// clang-format off
 /**
 * Performs element-wise hardtanh operation. The DST
 * register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available on the
 * compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | param0          | The minimum value of the linear region range                               | uint32_t |                                                       | True     |
 * | param1          | The maximum value of the linear region range                               | uint32_t |                                                       | True     |

 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void hardtanh_tile(uint32_t idst, uint32_t param0, uint32_t param1) {
    MATH((sfpu::Hardtanh<APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(idst, VectorMode::RC, param0, param1)));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void hardtanh_tile_pack(uint32_t idst, uint32_t param0, uint32_t param1) {
    PACK((sfpu::Hardtanh<APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(idst, VectorMode::RC, param0, param1)));
}

/**
 * Please refer to documentation for any_init.
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void hardtanh_tile_init() {
    MATH((sfpu::Hardtanh<APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void hardtanh_tile_init_pack() {
    PACK((sfpu::Hardtanh<APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

}  // namespace ckernel
