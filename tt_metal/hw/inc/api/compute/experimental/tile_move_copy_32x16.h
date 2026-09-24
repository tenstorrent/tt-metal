// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/tile_move_copy.h"

#ifdef TRISC_MATH
#include "experimental/llk_math_unary_datacopy_32x16_api.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Copies one 32x16 tile from an input CB directly into a 32x16 DEST slot. Unlike copy_tile(),
 * dst_tile_index addresses half-width DEST slots. Supported on Wormhole and Blackhole when both
 * the input and unpack destination formats are Int32 or Float32.
 *
 * Call copy_init(in_cb_id), wait for the input tile with cb_wait_front(), and acquire DEST before
 * calling this function.
 *
 * Return value: None
 *
 * | Argument       | Description                         | Type     | Valid range                                          | Required |
 * |----------------|-------------------------------------|----------|------------------------------------------------------|----------|
 * | in_cb_id       | Input circular buffer identifier    | uint32_t | 0 to 31                                              | True     |
 * | in_tile_index  | Tile index in the CB front          | uint32_t | Less than the number of tiles waited for             | True     |
 * | dst_tile_index | 32x16 destination tile index       | uint32_t | Less than the runtime 32x16 DEST tile capacity       | True     |
 */
// clang-format on
ALWI void copy_tile_to_dst_32x16(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index) {
    UNPACK((llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        in_cb_id, in_tile_index)));
    MATH((llk_math_eltwise_unary_datacopy_to_dest_32x16(dst_tile_index, in_cb_id)));
}

}  // namespace ckernel
