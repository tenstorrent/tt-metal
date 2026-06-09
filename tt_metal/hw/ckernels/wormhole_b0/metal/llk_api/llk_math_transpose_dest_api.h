// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_common_api.h"
#include "llk_math_transpose_dest.h"

/**
 * @brief Transpose a tile in place within the destination register.
 *
 * For 32-bit datums uses the format-switching path; otherwise runs the preconfigured transpose MOP.
 *
 * @tparam transpose_of_faces: Also transpose the arrangement of faces, not just elements within a face.
 * @tparam is_32bit: True for 32-bit datums (uses the format-switching transpose path).
 * @param dst_index: Tile index into the destination register.
 * @note Call @ref llk_math_transpose_dest_init with matching template args before this function.
 * @note On the unpack thread, the tile must already be in dest (via @ref llk_unpack_A datacopy);
 *       @ref llk_unpack_set_srcb_dummy_valid marks SrcB valid so the MOVB2D/MOVD2B sequence can run.
 * @note <transpose_of_faces=false, is_32bit=false> is not supported.
 */
template <bool transpose_of_faces = true, bool is_32bit = false>
inline void llk_math_transpose_dest(uint dst_index) {
    LLK_ASSERT((dst_index < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "");

    _llk_math_transpose_dest_<transpose_of_faces, is_32bit>(dst_index);
}

/**
 * @brief Configure the math thread (address mods and MOP) for a destination-register transpose.
 *
 * @tparam transpose_of_faces: Also transpose the arrangement of faces, not just elements within a face.
 * @tparam is_32bit: True for 32-bit datums (uses the format-switching transpose path).
 * @ref llk_math_transpose_dest runs the configured transpose with matching template args.
 */
template <bool transpose_of_faces = true, bool is_32bit = false>
inline void llk_math_transpose_dest_init() { _llk_math_transpose_dest_init_<transpose_of_faces, is_32bit>(); }
