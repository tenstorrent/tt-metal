// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_logsigmoid.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#if defined(ARCH_BLACKHOLE)
#include "ckernel_sfpu_logsigmoid_bf16.h"
#endif
#include "llk_math_eltwise_unary_sfpu_macros.h"
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
ALWI void logsigmoid_tile(uint32_t idst_in0, uint32_t idst_in1, uint32_t idst_out) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_logsigmoid,
        (APPROX, 8 /* ITERATIONS */),
        idst_in0,
        idst_in1,
        idst_out,
        VectorMode::RC)));
}

/**
 * Initialize logsigmoid operation.
 * Must be called before logsigmoid_tile.
 *
 * Return value: None
 */
ALWI void logsigmoid_tile_init() { MATH((SFPU_BINARY_INIT(unused))); }

// Blackhole only: ckernel_sfpu_logsigmoid_bf16.h exists under that ckernel tree.
// Wormhole and Quasar keep the op's own kernel.
#if defined(ARCH_BLACKHOLE)

// clang-format off
/**
 * Performs element-wise computation of log_sigmoid on each element of a tile in DEST, which holds
 * BF16 data, with one pass of a generated SFPU kernel. The DEST register buffer must be in acquired
 * state via *acquire_dst* call. This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void log_sigmoid_tile(uint32_t idst) {
    static_assert(!is_fp32_dest_acc_en, "log_sigmoid_tile evaluates BF16 DEST");
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_logsigmoid_bf16, (32), idst, VectorMode::None));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void log_sigmoid_tile_init() { MATH(SFPU_UNARY_INIT(unused)); }

#endif  // ARCH_BLACKHOLE

}  // namespace ckernel
