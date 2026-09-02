// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"

#ifdef TRISC_MATH
#include "llk_math_binary_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_AB_api.h"
#endif

namespace ckernel {

// ============================================================================
// Binary dest reuse add
// ============================================================================

/**
 * Init for binary dest reuse add
 */
template <EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::DEST_TO_SRCA>
ALWI void deepseek_binary_dest_reuse_add_tiles_init(uint32_t icb0, uint32_t call_line = __builtin_LINE()) {
    state_configure(icb0, call_line);
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, true, binary_reuse_dest>(false, false, icb0)));
    MATH(
        (llk_math_eltwise_binary_init<EltwiseBinaryType::ELWADD, BroadcastType::NONE, MATH_FIDELITY, binary_reuse_dest>(
            icb0, icb0)));
}

/**
 * Binary dest reuse add
 * dest[idst] = dest[idst] + cb[in_tile_index]
 */
template <
    bool fp32_dest_acc_en = DST_ACCUM_MODE,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::DEST_TO_SRCA>
ALWI void deepseek_binary_dest_reuse_add_tiles(uint32_t icb, uint32_t in_tile_index, uint32_t idst) {
    UNPACK((llk_unpack_A<BroadcastType::NONE, true, binary_reuse_dest>(icb, in_tile_index)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWADD,
          BroadcastType::NONE,
          fp32_dest_acc_en,
          MATH_FIDELITY,
          binary_reuse_dest>(icb, icb, idst, true)));
}

}  // namespace ckernel
