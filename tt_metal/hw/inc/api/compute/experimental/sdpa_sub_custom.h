// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"

#if defined(TRISC_MATH) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_math_eltwise_binary_custom_api.h"
#endif

#if defined(TRISC_UNPACK) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_unpack_AB_sub_bcast_col_custom_api.h"
#endif

namespace ckernel {

#ifdef ARCH_BLACKHOLE

ALWI void sub_bcast_cols_init_short_custom(uint32_t icb0, uint32_t icb1, uint32_t ct_dim, uint32_t call_line = __builtin_LINE()) {
    state_configure(icb0, icb1, call_line);
    MATH((llk_math_eltwise_binary_sub_bcast_cols_init_custom<MATH_FIDELITY>(icb0, icb1)));
    UNPACK((llk_unpack_AB_sub_bcast_col_init_custom<BroadcastType::COL>(icb0)));
}

ALWI void sub_tiles_bcast_cols_custom(
    uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst, uint32_t ct_dim) {
    MATH((llk_math_eltwise_binary_sub_bcast_cols_custom<DST_ACCUM_MODE>(idst, ct_dim)));
    UNPACK((llk_unpack_AB_sub_bcast_col_custom<BroadcastType::COL>(icb0, icb1, itile0, itile1, ct_dim)));
}

#endif

}  // namespace ckernel
