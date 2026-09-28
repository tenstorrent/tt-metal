// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "api/compute/common_globals.h"
#if defined(TT_POLY_LLK_DISABLE) || !__has_include(                                                                    \
                                        "ckernel_sfpu_multigammaln_bf16.h") || !defined(TT_POLY_BF16_UNARY_CONTEXT) || \
                                        TT_POLY_BF16_UNARY_CONTEXT != 1 || !defined(TT_METAL_SFPU_SINGLE_TILE_DST) ||  \
                                        TT_METAL_SFPU_SINGLE_TILE_DST != 1 || !defined(SFPU_OP_PROGRAM_INIT_0)
#error "Selected aggregate requires its enabled single-tile unary context"
#endif
#if defined(TRISC_MATH)
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_multigammaln_bf16.h"
#endif
namespace ckernel {
MATH(static_assert(DST_ACCUM_MODE, "Selected aggregate requires FP32 destination"));
ALWI void tt_poly_aggregate_multigammaln_tt_poly_bf16_program_init() {
    MATH(sfpu::init_tt_poly_aggregate_multigammaln_tt_poly_bf16());
}
// The kernel program hook owns initialization before the tile loop.
ALWI void tt_poly_aggregate_multigammaln_tt_poly_bf16_tile_init() {}
ALWI void tt_poly_aggregate_multigammaln_tt_poly_bf16_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_tt_poly_aggregate_multigammaln_tt_poly_bf16,
        (32),
        idst,
        VectorMode::None));
}
}  // namespace ckernel
