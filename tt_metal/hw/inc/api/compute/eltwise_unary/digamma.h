// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_digamma.h"
#endif

namespace ckernel {

ALWI void digamma_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_digamma, (APPROX), idst, VectorMode::RC));
}

ALWI void digamma_tile_init() { MATH(SFPU_UNARY_INIT_FN(unused, sfpu::digamma_init, (APPROX))); }

}  // namespace ckernel
