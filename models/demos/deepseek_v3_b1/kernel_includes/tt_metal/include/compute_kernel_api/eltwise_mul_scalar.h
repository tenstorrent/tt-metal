// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"

#ifdef TRISC_MATH
#include "llk_math_binary_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_AB_api.h"
#endif

namespace ckernel {

// ============================================================================
// Scalar broadcast multiply
// ============================================================================

/**
 * Hardware startup for scalar broadcast multiply.
 * Call once at kernel start. Same as compute_kernel_hw_startup() but with configurable fp32_dest_acc_en.
 */
template <bool fp32_dest_acc_en = false>
ALWI void deepseek_mul_tiles_bcast_scalar_hw_startup(uint32_t icb0, uint32_t icb1, uint32_t ocb) {
    UNPACK((llk_unpack_hw_configure<fp32_dest_acc_en>(icb0, icb1)));

    MATH((llk_math_pack_sync_init<fp32_dest_acc_en>()));
    MATH((llk_math_hw_configure<fp32_dest_acc_en>(icb0, icb1)));

    PACK((llk_pack_init<false, false, false>(ocb)));
    PACK((llk_pack_hw_configure<fp32_dest_acc_en>(ocb)));
    PACK((llk_pack_dest_init<fp32_dest_acc_en, false>(ocb)));
}

/**
 * Short init for scalar broadcast multiply (assumes hw already configured)
 */
ALWI void deepseek_mul_tiles_bcast_scalar_init_short(
    uint32_t icb0, uint32_t icb1, uint32_t call_line = __builtin_LINE()) {
    state_configure(icb0, icb1, call_line);
    MATH((llk_math_eltwise_binary_init_with_operands<ELWMUL, BroadcastType::SCALAR, MATH_FIDELITY>(icb0, icb1)));
    UNPACK((llk_unpack_AB_init<BroadcastType::SCALAR>(icb0, icb1)));
}

/**
 * Scalar broadcast multiply with configurable fp32 accumulation
 */
template <bool fp32_dest_acc_en = false>
ALWI void deepseek_mul_tiles_bcast_scalar(
    uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    MATH((llk_math_eltwise_binary<
          ELWMUL,
          BroadcastType::SCALAR,
          fp32_dest_acc_en,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true)));
    UNPACK((llk_unpack_AB<BroadcastType::SCALAR>(icb0, icb1, itile0, itile1)));
}

// ============================================================================
// Binary dest reuse multiply
// ============================================================================

/**
 * Init for binary dest reuse multiply
 */
template <
    bool fp32_dest_acc_en = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::DEST_TO_SRCA>
ALWI void deepseek_binary_dest_reuse_tiles_init(uint32_t icb0, uint32_t call_line = __builtin_LINE()) {
    state_configure(icb0, call_line);
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, fp32_dest_acc_en, binary_reuse_dest>(false, false, icb0)));
    MATH((llk_math_eltwise_binary_init<ELWMUL, BroadcastType::NONE, MATH_FIDELITY, binary_reuse_dest>(false)));
}

/**
 * Binary dest reuse multiply
 * dest[idst] = dest[idst] * cb[in_tile_index]
 */
template <
    bool fp32_dest_acc_en = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::DEST_TO_SRCA>
ALWI void deepseek_binary_dest_reuse_tiles(uint32_t icb, uint32_t in_tile_index, uint32_t idst) {
    UNPACK((llk_unpack_A<BroadcastType::NONE, fp32_dest_acc_en, binary_reuse_dest>(icb, in_tile_index)));
    MATH((llk_math_eltwise_binary<ELWMUL, BroadcastType::NONE, fp32_dest_acc_en, MATH_FIDELITY, binary_reuse_dest>(
        icb, icb, idst, true)));
}

}  // namespace ckernel
