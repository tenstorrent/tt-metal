// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_ldexp.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif

namespace ckernel {

// odst = idst0 * 2^idst1
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void ldexp_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_sfpu_ldexp,
        (APPROX, is_fp32_dest_acc_en, 8 /* ITERATIONS */),
        idst0,
        idst1,
        odst,
        VectorMode::RC)));
}

ALWI void ldexp_binary_tile_init() { MATH((SFPU_BINARY_INIT_FN_NO_ARGS(unused, sfpu::calculate_sfpu_ldexp_init))); }

}  // namespace ckernel
