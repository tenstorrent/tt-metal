// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_tanhshrink.h"
#endif

namespace ckernel {

ALWI void tanhshrink_tile(uint32_t idst) {
    MATH(SFPU_CALL_MODE(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_tanhshrink, (DST_ACCUM_MODE, 8 /* ITERATIONS */), RC, idst));
}

ALWI void tanhshrink_tile_init() { MATH(SFPU_INIT_CB(unused, sfpu::tanhshrink_init, (APPROX, DST_ACCUM_MODE))); }

}  // namespace ckernel
