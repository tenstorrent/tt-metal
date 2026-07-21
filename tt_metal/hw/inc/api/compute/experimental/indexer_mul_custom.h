// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"

// Blackhole-only: the math LLK lives only in the Blackhole llk_lib and indexer_score is a BH op.
#if defined(TRISC_MATH) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_math_eltwise_binary_custom_api.h"
#endif

#if defined(TRISC_UNPACK) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_unpack_AB_sub_bcast_col_custom_api.h"
#endif

namespace ckernel {

#if defined(ARCH_BLACKHOLE)

// Blocked bcast-col MUL with dest-MAC head reduction (indexer_score gate-mul). Mechanism documented
// canonically in _llk_math_bcast_cols_reuse_custom_ (llk_math_eltwise_binary_custom.h). Here: one
// unpack context loads ONE SrcB (gate w[h]) + ct_dim SrcA tiles (qk for ct_dim consecutive cols of
// head h, so cb_qk must be head-major). Reuses the SDPA blocked-sub unpack (op-agnostic). Pair with
// mul_bcast_cols_init_short_custom.

ALWI void mul_bcast_cols_init_short_custom(uint32_t icb0, uint32_t icb1, uint32_t call_line = __builtin_LINE()) {
    state_configure(icb0, icb1, call_line);
    MATH((llk_math_eltwise_binary_mul_bcast_cols_init_custom(icb0, icb1)));
    UNPACK((llk_unpack_AB_sub_bcast_col_init_custom(icb0)));
}

// itile0 = base SrcA (qk) tile of head h's column run (ct_dim consecutive tiles); itile1 = w[h].
ALWI void mul_tiles_bcast_cols_custom(
    uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst, uint32_t ct_dim) {
    MATH((llk_math_eltwise_binary_mul_bcast_cols_custom(idst, ct_dim)));
    UNPACK((llk_unpack_AB_sub_bcast_col_custom(icb0, icb1, itile0, itile1, ct_dim)));
}

#endif

}  // namespace ckernel
