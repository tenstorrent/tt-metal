// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_common_api.h"
#include "experimental/llk_math_eltwise_unary_datacopy_custom.h"

/*************************************************************************
 * LLK ELTWISE UNARY DATACOPY
 *************************************************************************/

template <
    DataCopyType type,
    bool is_fp32_dest_acc_en,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy_custom(uint dst_index, uint operand = 0) {
    LLK_ASSERT((dst_index < get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, DstTileShape::Tile32x32>()), "");

    const std::uint32_t operand_id = get_operand_id(operand);
    _llk_math_eltwise_unary_datacopy_custom_<
        type,
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        src_b_bcast_type,
        unpack_to_dest>(dst_index, unpack_src_format[operand_id], unpack_dst_format[operand_id]);
}
