// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"

#if defined(TRISC_MATH) && (defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE))
#include "experimental/llk_math_eltwise_binary_custom_api.h"
#endif

#if defined(TRISC_UNPACK) && (defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE))
#include "experimental/llk_unpack_AB_sub_bcast_col_custom_api.h"
#endif

namespace ckernel {

#if defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE)

// Blocked bcast-col MUL with dest-MAC head reduction (indexer_score gate-mul). One unpack context
// loads ONE SrcB (the gate w[h]) + ct_dim SrcA tiles (qk for ct_dim consecutive columns of head h,
// so cb_qk must be head-major), then ct_dim LoFi ELWMUL-MACs land column j on dest[idst + j]. The
// caller does ONE tile_regs_acquire for the column-batch and loops the heads (each head = one call,
// same idst), so dest[idst + j] = sum_h qk[col j, h] * w[h] -- one pack per column instead of one
// pack per (column, head). Reuses the proven SDPA blocked sub unpack (op-agnostic). The init forces
// both unpackers to stream whole tiles; pair with mul_bcast_cols_init_short_custom.

ALWI void mul_bcast_cols_init_short_custom(uint32_t icb0, uint32_t icb1, uint32_t call_line = __builtin_LINE()) {
    state_configure(icb0, icb1, call_line);
    MATH((llk_math_eltwise_binary_mul_bcast_cols_init_custom<MATH_FIDELITY>(icb0, icb1)));
    UNPACK((llk_unpack_AB_sub_bcast_col_init_custom(icb0)));
}

// itile0 = base SrcA (qk) tile of head h's column run (ct_dim consecutive tiles); itile1 = w[h].
ALWI void mul_tiles_bcast_cols_custom(
    uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst, uint32_t ct_dim) {
    MATH((llk_math_eltwise_binary_mul_bcast_cols_custom<DST_ACCUM_MODE>(idst, ct_dim)));
    UNPACK((llk_unpack_AB_sub_bcast_col_custom(icb0, icb1, itile0, itile1, ct_dim)));
}

#endif

}  // namespace ckernel
