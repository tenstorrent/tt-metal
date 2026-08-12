// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "experimental/llk_math_eltwise_unary_datacopy_softmax_k.h"
#include "llk_math_common_api.h"

/*************************************************************************
 * LLK ELTWISE UNARY DATACOPY — SOFTMAX K
 *************************************************************************/

inline void llk_math_eltwise_unary_datacopy_softmax_k(uint dst_index) {
    LLK_ASSERT((dst_index < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "");
    _llk_math_eltwise_unary_datacopy_softmax_k_(dst_index);
}

inline void llk_math_eltwise_unary_datacopy_softmax_k_init() { _llk_math_eltwise_unary_datacopy_softmax_k_init_(); }
