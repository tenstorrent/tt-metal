// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_unary_datacopy_api.h"
#include "experimental/llk_math_fast_tilize.h"

/*************************************************************************
 * LLK MATH FAST TILIZE (BH)
 *************************************************************************/

inline void llk_math_fast_tilize_init(const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    _llk_math_fast_tilize_init_<DST_ACCUM_MODE>(unpack_dst_format[operand_id]);
}

template <bool is_fp32_dest_acc_en>
inline void llk_math_fast_tilize_uninit(const std::uint32_t operand) {
    const std::uint32_t operand_id = get_operand_id(operand);
    _llk_math_fast_tilize_uninit_<is_fp32_dest_acc_en>(unpack_dst_format[operand_id]);
}

inline void llk_math_fast_tilize_block_(
    const std::uint32_t dst_index, const std::uint32_t operand, const std::uint32_t unit_dim) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    _llk_math_fast_tilize_block_<DST_ACCUM_MODE>(dst_index, unpack_dst_format[operand_id], unit_dim, num_faces);
}
