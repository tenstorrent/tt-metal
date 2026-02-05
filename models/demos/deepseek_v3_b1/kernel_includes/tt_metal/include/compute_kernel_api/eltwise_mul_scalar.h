// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/bcast.h"

#ifdef TRISC_MATH
#include "llk_math_binary_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_AB_api.h"
#endif

namespace ckernel {

// ============================================================================
// Scalar broadcast multiply with fp32 accumulation
// ============================================================================

/**
 * Init for scalar broadcast multiply
 */
ALWI void mul_tiles_bcast_scalar_init_short_fp32(uint32_t icb0, uint32_t icb1, uint32_t call_line = __builtin_LINE()) {
    state_configure(icb0, icb1, call_line);
    MATH((llk_math_eltwise_binary_init_with_operands<ELWMUL, BroadcastType::SCALAR, MATH_FIDELITY>(icb0, icb1)));
    UNPACK((llk_unpack_AB_init<BroadcastType::SCALAR>(icb0, icb1)));
}

/**
 * Scalar broadcast multiply with fp32 accumulation
 */
ALWI void mul_tiles_bcast_scalar_fp32(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    MATH((llk_math_eltwise_binary<ELWMUL, BroadcastType::SCALAR, true, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(
        icb0, icb1, idst, true)));
    UNPACK((llk_unpack_AB<BroadcastType::SCALAR>(icb0, icb1, itile0, itile1)));
}

// ============================================================================
// Binary dest reuse multiply with fp32 accumulation
// ============================================================================

/**
 * Init for binary dest reuse multiply
 */
template <EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::DEST_TO_SRCA>
ALWI void binary_dest_reuse_tiles_init_fp32(uint32_t icb0, uint32_t call_line = __builtin_LINE()) {
    state_configure(icb0, call_line);
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, true, binary_reuse_dest>(false, false, icb0)));
    MATH((llk_math_eltwise_binary_init<ELWMUL, BroadcastType::NONE, MATH_FIDELITY, binary_reuse_dest>(false)));
}

/**
 * Binary dest reuse multiply with fp32 accumulation
 * dest[idst] = dest[idst] * cb[in_tile_index]
 */
template <EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::DEST_TO_SRCA>
ALWI void binary_dest_reuse_tiles_fp32(uint32_t icb, uint32_t in_tile_index, uint32_t idst) {
    UNPACK((llk_unpack_A<BroadcastType::NONE, true, binary_reuse_dest>(icb, in_tile_index)));
    MATH((llk_math_eltwise_binary<ELWMUL, BroadcastType::NONE, true, MATH_FIDELITY, binary_reuse_dest>(
        icb, icb, idst, true)));
}

}  // namespace ckernel
