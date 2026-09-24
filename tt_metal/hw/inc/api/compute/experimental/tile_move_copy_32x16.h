// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/tile_move_copy.h"

#ifdef TRISC_MATH
#include "experimental/llk_math_unary_datacopy_32x16_api.h"
#endif

namespace ckernel {

ALWI void copy_tile_to_dst_32x16(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index) {
    UNPACK((llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        in_cb_id, in_tile_index)));
    MATH((llk_math_eltwise_unary_datacopy_to_dest_32x16(dst_tile_index, in_cb_id)));
}

}  // namespace ckernel
