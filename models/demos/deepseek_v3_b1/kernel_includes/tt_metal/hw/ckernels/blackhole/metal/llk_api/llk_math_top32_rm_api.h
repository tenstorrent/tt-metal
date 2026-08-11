// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "../../../../../../tt_llk/tt_llk_blackhole/llk_lib/llk_math_top32_rm.h"
#include "llk_math_common_api.h"
#include "llk_math_transpose_dest_api.h"

/*****************************************
 * LLK MATH — Top32 row-major transpose
 *****************************************/

inline void llk_math_top32_rm_init(const std::uint32_t icb) {
    const std::uint32_t operand_id = get_operand_id(icb);
    _llk_math_top32_rm_init_<DST_ACCUM_MODE>(get_operand_num_faces(operand_id), get_operand_dst_format(operand_id));

    const std::uint32_t src_format = get_operand_src_format(operand_id);
    const bool is_int32 = (src_format & 0xf) == (std::uint32_t)DataFormat::Int32;
    if (is_int32) {
        llk_math_transpose_dest_init<false, true>();
    }
}

inline void llk_math_top32_rm(const std::uint32_t icb, const std::uint32_t idst, const std::uint32_t num_faces) {
    const std::uint32_t operand_id = get_operand_id(icb);
    const std::uint32_t src_format = get_operand_src_format(operand_id);
    const std::uint32_t dst_format = get_operand_dst_format(operand_id);
    const bool is_int32 = (src_format & 0xf) == (std::uint32_t)DataFormat::Int32;

    if (is_int32) {
        _llk_math_top32_rm_<DST_SYNC_MODE, DST_ACCUM_MODE, UnpackToDestEn>(idst, src_format, dst_format, num_faces);
        llk_math_transpose_dest<false, true>(idst);
    } else {
        _llk_math_top32_rm_<DST_SYNC_MODE, DST_ACCUM_MODE, false>(idst, src_format, dst_format, num_faces);
    }
}
