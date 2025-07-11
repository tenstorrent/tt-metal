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
#if defined(TRISC_MATH) || defined(TRISC_UNPACK)
    const std::uint32_t src_format = get_operand_src_format(icb);
    const bool is_int32 = (src_format & 0xf) == (std::uint32_t)DataFormat::Int32;

    if (is_int32) {
        UNPACK((llk_unpack_A_init<
                BroadcastType::NONE /*BType*/,
                false /*acc_to_dest*/,
                EltwiseBinaryReuseDestType::NONE /*binary_reuse_dest*/,
                UnpackToDestEn /*unpack_to_dest*/,
                true /*disable_src_zero_flag*/>(true /*transpose_of_faces*/, false /*within_face_16x16_transpose*/)));
        MATH((llk_math_eltwise_unary_datacopy_init<
              A2D /*type*/,
              DST_ACCUM_MODE /*is_fp32_dest_acc_en*/,
              BroadcastType::NONE /*BType*/>(
            true /*transpose_of_faces*/, false /*within_face_16x16_transpose*/, icb /*operand*/)));
        MATH((llk_math_transpose_dest_init<false /*transpose_of_faces*/, true /*is_32bit*/>()));
    } else {
        UNPACK((llk_unpack_A_init<
                BroadcastType::NONE /*Btype*/,
                true /*acc_to_dest*/,
                EltwiseBinaryReuseDestType::NONE /*binary_reuse_dest*/>(
            true /*transpose_of_faces*/, true /*within_face_16x16_transpose*/)));
        MATH((llk_math_eltwise_unary_datacopy_init<
              A2D /*type*/,
              DST_ACCUM_MODE /*is_fp32_dest_acc_en*/,
              BroadcastType::NONE /*BType*/>(
            true /*transpose_of_faces*/, true /*within_face_16x16_transpose*/, icb /*operand*/)));
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
#if defined(TRISC_MATH) || defined(TRISC_UNPACK)
    const std::uint32_t src_format = get_operand_src_format(icb);
    const bool is_int32 = (src_format & 0xf) == (std::uint32_t)DataFormat::Int32;

    if (is_int32) {
        UNPACK((llk_unpack_A<
                BroadcastType::NONE /*Btype*/,
                false /*acc_to_dest*/,
                EltwiseBinaryReuseDestType::NONE /*binary_reuse_dest*/,
                UnpackToDestEn /*unpack_to_dest*/>(
            icb /*operand*/, itile /*tile_index*/, true /*transpose_of_faces*/)));
        UNPACK((llk_unpack_set_srcb_dummy_valid()));
        MATH((llk_math_eltwise_unary_datacopy<
              A2D /*type*/,
              DST_ACCUM_MODE /*is_fp32_dest_acc_en*/,
              BroadcastType::NONE /*BType*/,
              UnpackToDestEn /*unpack_to_dest*/>(idst /*dst_tile_index*/)));
        MATH((llk_math_transpose_dest<false /*transpose_of_faces*/, true /*is_32bit*/>(idst /*dst_tile_index*/)));
    } else {
        UNPACK((llk_unpack_A<
                BroadcastType::NONE /*Btype*/,
                false /*acc_to_dest*/,
                EltwiseBinaryReuseDestType::NONE /*binary_reuse_dest*/,
                UnpackToDestEn /*unpack_to_dest*/>(
            icb /*operand*/, itile /*tile_index*/, true /*transpose_of_faces*/)));
        MATH((llk_math_eltwise_unary_datacopy<
              A2D /*type*/,
              DST_ACCUM_MODE /*is_fp32_dest_acc_en*/,
              BroadcastType::NONE /*BType*/>(idst /*dst_tile_index*/)));
    }
#endif
}

}  // namespace ckernel
