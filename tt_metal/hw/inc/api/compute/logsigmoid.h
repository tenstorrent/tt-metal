// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_logsigmoid.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#if defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)
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
 * On Wormhole this loads polynomial coefficients into vConstFloatPrgm0..2, the registers exp_tile_init
 * (and other SFPU inits) also program. Call it after those inits, right before logsigmoid_tile, and call
 * exp_tile_init again after it before the next exp_tile.
 *
 * Return value: None
 */
ALWI void logsigmoid_tile_init() { MATH((SFPU_BINARY_INIT_FN(unused, sfpu::logsigmoid_init, (APPROX)))); }

// The generated kernel exists on Blackhole and Wormhole, where TT-NN runs it; Quasar keeps the op's own kernel, and
// a call there fails to build.
// clang-format off
/**
 * Performs element-wise computation of log_sigmoid on each element of a tile in DEST, which holds
 * BF16 data, with one pass of a generated SFPU kernel. The DEST register buffer must be in acquired
 * state via *acquire_dst* call. This call is blocking and is only available on the compute engine.
 *
 * It computes ttnn.log_sigmoid of the tile's own values in this one call. `logsigmoid_tile`, above, is a step of TT-NN's own log_sigmoid kernel, which takes other operands and is unchanged.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE, bool on_board = false>
ALWI void log_sigmoid_tile([[maybe_unused]] uint32_t idst) {
#if defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)
    static_assert(!is_fp32_dest_acc_en, "log_sigmoid_tile evaluates BF16 DEST");
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_logsigmoid_bf16, (32), idst, VectorMode::None));
#else
    static_assert(on_board, "log_sigmoid_tile: Blackhole and Wormhole only");
#endif
}

/**
 * Please refer to documentation for any_init.
 */
template <bool on_board = false>
ALWI void log_sigmoid_tile_init() {
#if defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)
    MATH(SFPU_UNARY_INIT(unused));
    MATH(sfpu::init_logsigmoid_bf16());
#else
    static_assert(on_board, "log_sigmoid_tile_init: Blackhole and Wormhole only");
#endif
}

}  // namespace ckernel
