// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#include "api/compute/tile_move_copy.h"

#ifdef TRISC_MATH
#include "llk_math_unary_datacopy_api.h"
#endif

#ifdef TRISC_UNPACK
#include "experimental/llk_unpack_A_src_safe_custom_api.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Src-safe variant of copy_tile_to_dst_init_short. The init path is unchanged
 * from the production version. It is provided here only so callers can pair
 * init and exec with matching _src_safe suffixes. Delegates to
 * copy_tile_to_dst_init_short. Wormhole B0 only; failing to resolve on other
 * archs is the desired signal that this experimental API is WH B0 only.
 */
// clang-format on
ALWI void copy_tile_src_safe_init(
    uint32_t cbid,
    uint32_t transpose = 0,
    uint32_t transpose_within_16x16_face = false,
    uint32_t call_line = __builtin_LINE()) {
    copy_tile_to_dst_init_short(cbid, transpose, transpose_within_16x16_face, call_line);
}

// clang-format off
/**
 * Src-safe variant of copy_tile. Identical to copy_tile for all non-32-bit
 * unpack-to-dest paths; on INT32 / UInt32 unpack-to-dest it avoids the
 * TEN-3868 workaround's UndefinedBehavior by temporarily swapping in/out
 * formats to UInt16 around the dummy SrcA unpack issued after the last
 * unpack-to-dest tile. Wormhole B0 only.
 */
// clang-format on
ALWI void copy_tile_src_safe(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index) {
    UNPACK((llk_unpack_A_src_safe_custom<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        in_cb_id, in_tile_index)));
    MATH((llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
        dst_tile_index, in_cb_id)));
}

// clang-format off
/**
 * Src-safe variant of copy_block_matmul_partials. Wormhole B0 only.
 */
// clang-format on
ALWI void copy_block_matmul_partials_src_safe(
    uint32_t in_cb_id, uint32_t start_in_tile_index, uint32_t start_dst_tile_index, uint32_t ntiles) {
    UNPACK((llk_unpack_A_block_src_safe_custom<
            BroadcastType::NONE,
            false,
            EltwiseBinaryReuseDestType::NONE,
            UnpackToDestEn>(in_cb_id, start_in_tile_index, ntiles)));
    MATH((llk_math_eltwise_unary_datacopy_block<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
        start_dst_tile_index, ntiles, in_cb_id)));
}

}  // namespace ckernel
