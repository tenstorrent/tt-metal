// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "experimental/llk_math_eltwise_unary_datacopy_topk_xl_copy.h"
#include "llk_math_common_api.h"

/*************************************************************************
 * LLK MATH — TopK-XL copy (single-outer-loop A2D MOP + stock datacopy execute path)
 *************************************************************************/

inline void llk_math_topk_xl_copy_init(const std::uint32_t operand = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t dst_format = get_operand_dst_format(operand_id);
    ckernel::_llk_math_topk_xl_copy_init_(dst_format);
}

inline void llk_math_topk_xl_copy_one_tile_math(
    const std::uint32_t operand, const std::uint32_t dst_tile_index, const std::uint32_t elements_this_tile) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t dst_format = get_operand_dst_format(operand_id);
    ckernel::_llk_math_topk_xl_copy_(dst_tile_index, dst_format, elements_this_tile);
}
