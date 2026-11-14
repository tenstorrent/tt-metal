// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"

// The generated kernel exists on Blackhole and Wormhole, where TT-NN runs it; Quasar keeps the op's own kernel, and
// a call there fails to build.
#if defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)
#ifdef TRISC_MATH
#include "ckernel_sfpu_hardswish_bf16.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise computation of hardswish on each element of a tile in DEST, which holds
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
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE, bool on_board = false>
ALWI void hardswish_tile([[maybe_unused]] uint32_t idst) {
#if defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)
    static_assert(!is_fp32_dest_acc_en, "hardswish_tile evaluates BF16 DEST");
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_hardswish_bf16, (32), idst, VectorMode::None));
#else
    static_assert(on_board, "hardswish_tile: Blackhole and Wormhole only");
#endif
}

/**
 * Please refer to documentation for any_init.
 */
template <bool on_board = false>
ALWI void hardswish_tile_init() {
#if defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)
    MATH(SFPU_UNARY_INIT(unused));
    MATH(sfpu::init_hardswish_bf16());
#else
    static_assert(on_board, "hardswish_tile_init: Blackhole and Wormhole only");
#endif
}

}  // namespace ckernel
