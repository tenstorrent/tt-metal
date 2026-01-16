// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
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

// clang-format off
/**
 * Paired Init function for transpose_wh. For general information on init functions refer to any_init.
 *
 * | Argument       | Description                                                 | Type     | Valid Range | Required |
 * |----------------|-------------------------------------------------------------|----------|-------------|----------|
 * | icb            | The identifier of the circular buffer (CB) containing input | uint32_t | 0 to 31     | True     |
 */
 // clang-format on
ALWI void transpose_wh_init(uint32_t icb, uint32_t ocb) {

#if defined(TRISC_MATH) || defined(TRISC_UNPACK)
    const std::uint32_t src_format = get_operand_src_format(icb);
    const bool is_int32 = (src_format & 0xf) == (std::uint32_t)DataFormat::Int32;

    if (is_int32) {
        UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE, true>(icb)));
        UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
            true, false, icb)));
        MATH((llk_math_eltwise_unary_datacopy_init<A2D, DST_ACCUM_MODE, BroadcastType::NONE>(icb)));
        MATH((llk_math_transpose_dest_init<false, true>()));
    } else {
        UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(icb)));
        UNPACK((llk_unpack_A_init<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::NONE>(true, true, icb)));
        MATH((llk_math_eltwise_unary_datacopy_init<A2D, DST_ACCUM_MODE, BroadcastType::NONE>(icb)));
    }
    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(icb, icb)));
#endif

    PACK((llk_pack_hw_configure_disaggregated<DST_ACCUM_MODE, false>(ocb)));
    PACK((llk_pack_init(ocb)));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, false>()));
}

/**
 * Performs a first-call or switch-from-another-op tile hw reconfiguration step needed for transpose_wh to be executed
 * correctly.
 */
ALWI void transpose_wh_init_short(uint32_t icb) {
#if defined(TRISC_MATH) || defined(TRISC_UNPACK)
    const std::uint32_t src_format = get_operand_src_format(icb);
    const bool is_int32 = (src_format & 0xf) == (std::uint32_t)DataFormat::Int32;

    if (is_int32) {
        UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
            true, false, icb)));
        MATH((llk_math_eltwise_unary_datacopy_init<A2D, DST_ACCUM_MODE, BroadcastType::NONE>(icb)));
        MATH((llk_math_transpose_dest_init<false, true>()));
    } else {
        UNPACK((llk_unpack_A_init<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::NONE>(true, true, icb)));
        MATH((llk_math_eltwise_unary_datacopy_init<A2D, DST_ACCUM_MODE, BroadcastType::NONE>(icb)));
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
ALWI void transpose_wh_tile(uint32_t icb, uint32_t itile, uint32_t idst) {
#if defined(TRISC_MATH) || defined(TRISC_UNPACK)
    const std::uint32_t src_format = get_operand_src_format(icb);
    const bool is_int32 = (src_format & 0xf) == (std::uint32_t)DataFormat::Int32;

    if (is_int32) {
        UNPACK(
            (llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(icb, itile)));
        UNPACK((llk_unpack_set_srcb_dummy_valid()));
        MATH((llk_math_eltwise_unary_datacopy<A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(idst, icb)));
        MATH((llk_math_transpose_dest<false, true>(idst)));
    } else {
        UNPACK((llk_unpack_A<BroadcastType::NONE, false>(icb, itile)));
        MATH((llk_math_eltwise_unary_datacopy<A2D, DST_ACCUM_MODE, BroadcastType::NONE>(idst, icb)));
    }
#endif
}

}  // namespace ckernel
