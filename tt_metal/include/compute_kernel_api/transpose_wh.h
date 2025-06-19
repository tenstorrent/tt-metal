// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common.h"
#ifdef TRISC_MATH
#include "llk_math_common_api.h"
#include "llk_math_unary_datacopy_api.h"
#include "llk_math_transpose_dest_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_common_api.h"
#include "llk_unpack_A_api.h"
#endif

namespace ckernel {

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for transpose op to be executed
 * correctly.
 */
ALWI void transpose_init(uint32_t icb) {
    MATH((llk_math_eltwise_unary_datacopy_init<A2D, DST_ACCUM_MODE, BroadcastType::NONE>(true, true, icb)));

    if (is_int32) {
        UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(true, false)));
        MATH((llk_math_eltwise_unary_datacopy_init<A2D, DST_ACCUM_MODE, BroadcastType::NONE>(true, false, icb)));
        MATH((llk_math_transpose_dest_init<false, true>()));
    } else {
        UNPACK((llk_unpack_A_init<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::NONE>(true, true)));
        MATH((llk_math_eltwise_unary_datacopy_init<A2D, DST_ACCUM_MODE, BroadcastType::NONE>(true, true, icb)));
    }

#endif
}

// clang-format off
/**
 * Performs a 32x32 transpose operation *B[w,h] = A[h,w]* on a tile in the CB
 * at a given index and writes the result to the DST register at index
 * dst_tile_index. The DST register buffer must be in acquired state via
 * *acquire_dst* call.
 *
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                             | Type     | Valid Range                                    | Required |
 * |----------------|---------------------------------------------------------|----------|------------------------------------------------|----------|
 * | in_cb_id       | The identifier of the circular buffer (CB) containing A | uint32_t | 0 to 31                                        | True     |
 * | in_tile_index  | The index of tile A within the first CB                 | uint32_t | Must be less than the size of the CB           | True     |
 * | dst_tile_index | The index of the tile in DST REG for the result B       | uint32_t | Must be less than the acquired size of DST REG | True     |
 */
 // clang-format on
ALWI void transpose_tile(uint32_t icb, uint32_t itile, uint32_t idst) {
    UNPACK((llk_unpack_A<BroadcastType::NONE, false>(icb, itile, false)));
    MATH((llk_math_eltwise_unary_datacopy<A2D, DST_ACCUM_MODE, BroadcastType::NONE>(idst)));
}

}  // namespace ckernel
