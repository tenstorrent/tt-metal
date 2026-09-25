// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "api/compute/common_globals.h"
#if defined(TT_POLY_LLK_DISABLE) || (!defined(ARCH_BLACKHOLE) && !defined(ARCH_WORMHOLE)) || DST_ACCUM_MODE || \
    !defined(TT_POLY_BINARY_GRADIENT_CONTEXT) || TT_POLY_BINARY_GRADIENT_CONTEXT != 1
#error "Selected backward factor requires its enabled BF16 single-tile binary route"
#endif
#if defined(TRISC_MATH)
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_tanhshrink_bw_bf16.h"
#endif
namespace ckernel {
MATH(static_assert(!DST_ACCUM_MODE, "Selected backward factor requires BF16 destination"));
ALWI void tanhshrink_bw_tt_poly_bf16_tile_init() {
    MATH(SFPU_UNARY_INIT(unused));
    MATH(sfpu::init_tanhshrink_bw_tt_poly_bf16());
}
ALWI void tanhshrink_bw_tt_poly_bf16_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_tanhshrink_bw_tt_poly_bf16, (32), idst, VectorMode::None));
    MATH(if constexpr (ttpoly_generated::TanhshrinkBwBf16Config::needs_gradient) {
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_tanhshrink_bw_gradient_tt_poly_bf16,
            (32),
            idst,
            VectorMode::None));
    });
}
ALWI void tanhshrink_bw_tt_poly_bf16_gradient(uint32_t lhs, uint32_t, uint32_t) {
    tanhshrink_bw_tt_poly_bf16_tile(lhs);
}
}  // namespace ckernel
