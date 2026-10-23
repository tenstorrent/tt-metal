// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"

// Blackhole and Wormhole only: ckernel_sfpu_multigammaln_bf16.h exists under those ckernel trees.
// Quasar keeps the composite.
#if defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)

#ifdef TRISC_MATH
#include "ckernel_sfpu_multigammaln_bf16.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs element-wise computation of multigammaln on each element of a tile in DEST, which holds
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
ALWI void multigammaln_tile(uint32_t idst) {
    static_assert(!is_fp32_dest_acc_en, "multigammaln_tile evaluates BF16 DEST");
    MATH(
        SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_multigammaln_bf16, (32), idst, VectorMode::None));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void multigammaln_tile_init() {
    MATH(SFPU_UNARY_INIT(unused));
    MATH(sfpu::init_multigammaln_bf16());
}

}  // namespace ckernel

#endif  // ARCH_BLACKHOLE || ARCH_WORMHOLE
