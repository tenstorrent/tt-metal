// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "api/compute/common_globals.h"
#if defined(TT_POLY_LLK_DISABLE) || (!defined(ARCH_BLACKHOLE) && !defined(ARCH_WORMHOLE)) || DST_ACCUM_MODE || \
    !defined(TT_POLY_WHERE_CONDITION_FACTOR_CONTEXT) || TT_POLY_WHERE_CONDITION_FACTOR_CONTEXT != 1
#error "Selected backward factor requires its enabled BF16 single-tile WHERE route"
#endif
#if defined(TRISC_MATH)
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_hardshrink_bw_bf16.h"
#endif
namespace ckernel {
MATH(static_assert(!DST_ACCUM_MODE, "Selected backward factor requires BF16 destination"));
ALWI void hardshrink_bw_tt_poly_bf16_tile_init() { MATH(SFPU_UNARY_INIT(unused)); }
ALWI void hardshrink_bw_tt_poly_bf16_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_hardshrink_bw_tt_poly_bf16, (32), idst, VectorMode::None));
}
}  // namespace ckernel
