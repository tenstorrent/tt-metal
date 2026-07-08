// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
#include "api/compute/sentinel/compute_kernel_sentinel.h"
#include "llk_assert.h"
#include "sanitizer/api.h"
#ifdef TRISC_MATH
#include "llk_math_common_api.h"
#include "llk_math_unary_datacopy_api.h"
#ifndef ARCH_QUASAR
#include "llk_math_transpose_dest_api.h"
#endif
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_common_api.h"
#include "llk_unpack_A_api.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Paired init function for transpose_tile. Must be preceded - exactly once, at the very top of the
 * kernel - by compute_kernel_hw_startup(icb, ocb), which performs the one-time hardware
 * configuration. transpose_init() then reconfigures the unpacker/math pipeline for the transpose op
 * and is the function to call before transpose_tile() (including when switching to transpose from
 * another op). For general information on init functions refer to any_init.
 *
 * | Argument | Description                                                 | Type     | Valid Range | Required |
 * |----------|-------------------------------------------------------------|----------|-------------|----------|
 * | icb      | The identifier of the circular buffer (CB) containing input | uint32_t | 0 to 31     | True     |
 */
// clang-format on
ALWI void transpose_init(uint32_t icb, uint32_t call_line = __builtin_LINE()) {
    LLK_SAN_FUNCTION();
    state_configure(icb, call_line);
#if defined(TRISC_MATH) || defined(TRISC_UNPACK)
    const std::uint32_t src_format = get_operand_src_format(icb);
    const std::uint32_t dst_format = get_operand_dst_format(icb);

#ifndef ARCH_QUASAR
    // Low-nibble compare intentionally matches both signed Int8 (14) and unsigned UInt8
    // (30 -> low nibble 0xE): both 8-bit integer formats need the int-FPU (ELWADD) A2D
    // reconstruct path.
    const bool is_8bit_int = (src_format & 0xf) == (std::uint32_t)DataFormat::Int8;
    const bool enable_unpack_to_dest = (dst_format == (std::uint32_t)DataFormat::Float32) ||
                                       (dst_format == (std::uint32_t)DataFormat::UInt32) ||
                                       (dst_format == (std::uint32_t)DataFormat::Int32);
    if (enable_unpack_to_dest) {
        UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
            true, false, icb)));
        MATH((llk_math_eltwise_unary_datacopy_init<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE>(icb)));
        MATH((llk_math_transpose_dest_init<false, true>()));
    } else if (is_8bit_int) {
        // 8-bit integer (Int8/UInt8) transpose needs the int-FPU (ELWADD) A2D reconstruct path,
        // selected here via is_int_fpu_en. Ideally the LLK layer would infer this path from the
        // data format instead of selecting it here in the Compute API layer.
        // TODO: #46832.
        UNPACK((llk_unpack_A_init<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::NONE>(true, true, icb)));
        MATH((llk_math_eltwise_unary_datacopy_init<
              DataCopyType::A2D,
              DST_ACCUM_MODE,
              BroadcastType::NONE,
              true /*is_int_fpu_en*/>(icb)));
    } else {
        UNPACK((llk_unpack_A_init<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::NONE>(true, true, icb)));
        MATH((llk_math_eltwise_unary_datacopy_init<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE>(icb)));
    }
#else
    // Quasar has no unpack-to-dest transpose path (TODO: tt-llk#1559) and no int-FPU 8-bit integer
    // reconstruct path; reject formats that would otherwise silently take the wrong path. UInt32 is
    // treated as unpack-to-dest on WH/BH and Int8/UInt8 (low nibble 0xE) need the int-FPU path.
    const bool is_8bit_int = (src_format & 0xf) == (std::uint32_t)DataFormat::Int8;
    const bool enable_unpack_to_dest = (dst_format == (std::uint32_t)DataFormat::Float32) ||
                                       (dst_format == (std::uint32_t)DataFormat::UInt32) ||
                                       (dst_format == (std::uint32_t)DataFormat::Int32);
    LLK_ASSERT(
        !enable_unpack_to_dest, "32-bit (unpack-to-dest) transpose not supported on Quasar");  // TODO: tt-llk#1559
    LLK_ASSERT(!is_8bit_int, "8-bit integer transpose not supported on Quasar");
    UNPACK((llk_unpack_A_init<
            BroadcastType::NONE,
            false /*acc_to_dest*/,
            EltwiseBinaryReuseDestType::NONE,
            false /*unpack_to_dest*/>(true /*transpose_of_faces*/, true /*within_face_16x16_transpose*/, icb)));
    MATH((llk_math_eltwise_unary_datacopy_init<ckernel::DataCopyType::A2D, DST_ACCUM_MODE>(icb)));
#endif

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
    LLK_SAN_FUNCTION();
#if defined(TRISC_MATH) || defined(TRISC_UNPACK)
    const std::uint32_t dst_format = get_operand_dst_format(icb);

#ifndef ARCH_QUASAR
    const bool enable_unpack_to_dest = (dst_format == (std::uint32_t)DataFormat::Float32) ||
                                       (dst_format == (std::uint32_t)DataFormat::UInt32) ||
                                       (dst_format == (std::uint32_t)DataFormat::Int32);
    if (enable_unpack_to_dest) {
        UNPACK(
            (llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(icb, itile)));
        UNPACK((llk_unpack_set_srcb_dummy_valid()));
        MATH((llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(idst, icb)));
        MATH((llk_math_transpose_dest<false, true>(idst)));
    } else {
        UNPACK((llk_unpack_A<BroadcastType::NONE, false>(icb, itile)));
        MATH((llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE>(idst, icb)));
    }
#else
    const bool enable_unpack_to_dest = (dst_format == (std::uint32_t)DataFormat::Float32) ||
                                       (dst_format == (std::uint32_t)DataFormat::UInt32) ||
                                       (dst_format == (std::uint32_t)DataFormat::Int32);
    LLK_ASSERT(
        !enable_unpack_to_dest, "32-bit (unpack-to-dest) transpose not supported on Quasar");  // TODO: tt-llk#1559
    UNPACK((llk_unpack_A<BroadcastType::NONE, false /*acc_to_dest*/>(icb, itile)));
    MATH((llk_math_eltwise_unary_datacopy(idst, icb)));
#endif
#endif
}

}  // namespace ckernel
