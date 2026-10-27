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

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void tanhshrink_tile(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_tanhshrink,
        (is_fp32_dest_acc_en, 8 /* ITERATIONS */),
        idst,
        VectorMode::RC));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void tanhshrink_tile_init() { MATH(SFPU_UNARY_INIT_FN(unused, sfpu::tanhshrink_init, (APPROX, is_fp32_dest_acc_en))); }

}  // namespace ckernel
