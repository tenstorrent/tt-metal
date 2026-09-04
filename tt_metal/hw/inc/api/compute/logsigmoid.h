// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_logsigmoid.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs logsigmoid operation: logsigmoid(x) = -softplus(-x) = -log(1 + exp(-x))
 *
 * Return value: None
 *
 * | Argument       | Description                                       | Type     | Valid Range                                           | Required |
 * |----------------|---------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst_in0       | Index of tile in DST with input (x)               | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst_in1       | Index of tile in DST with exp(-x)                 | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst_out       | Index of tile in DST for output                   | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void logsigmoid_tile(uint32_t idst_in0, uint32_t idst_in1, uint32_t idst_out) {
    MATH((sfpu::Logsigmoid<APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst_in0, idst_in1, idst_out, VectorMode::RC)));
}

/**
 * Initialize logsigmoid operation.
 * Must be called before logsigmoid_tile.
 *
 * Return value: None
 */
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void logsigmoid_tile_init() {
    MATH((sfpu::Logsigmoid<APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

}  // namespace ckernel
