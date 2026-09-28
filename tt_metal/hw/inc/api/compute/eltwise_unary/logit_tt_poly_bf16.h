// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "api/compute/common_globals.h"
#if !defined(TT_POLY_LLK_DISABLE) && __has_include("ckernel_sfpu_logit_bf16.h")
#if defined(TRISC_MATH)
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_logit_bf16.h"
#endif
namespace ckernel {
// Internal entry: the source-owned composite kernel guards dtype and precision.
ALWI void logit_tt_poly_bf16_tile_init() {}
ALWI void logit_tt_poly_bf16_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_logit_tt_poly_bf16, (32 /* ITERATIONS */), idst, VectorMode::None));
}
}  // namespace ckernel
#endif
