// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "llk_math_unary_datacopy_api.h"  // legacy CB-id API + unified datacopy impls
#include "data_format_derive.h"
#include "api/compute/experimental/2_0/llk_mem_descriptor.h"

/*************************************************************************
 * LLK ELTWISE UNARY DATACOPY -- LLKOperand (id-free, compile-time NTTP) overloads
 *
 * Same function names as the CB-id API (llk_math_eltwise_unary_datacopy / _init), distinguished by
 * taking an LLKMemDescriptor as the first NTTP. MATH never touches L1, so DESC (format + geometry) is all
 * it consumes; the register format is derived HERE from DESC.format + the fp32-dest-acc flag and never
 * exposed above the LLK. Both overloads forward to the same unified cores as the CB-id API.
 *************************************************************************/

template <
    ckernel::experimental::LLKMemDescriptor DESC,
    DataCopyType type = DataCopyType::A2D,
    bool is_fp32_dest_acc_en = false,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool is_int_fpu_en = false,
    PackMode pack_mode = PackMode::Default>
inline void llk_math_eltwise_unary_datacopy_init() {
    constexpr std::uint8_t RegFmt = static_cast<std::uint8_t>(
        ckernel::infer_unpack_dst_format(static_cast<DataFormat>(DESC.format), is_fp32_dest_acc_en));
    // 8-bit input datums do not require the tilize workaround on Blackhole (folds to a constant).
    const bool is_input_8bit_format = IS_8BIT_FORMAT(DESC.format);
    llk_math_eltwise_unary_datacopy_init_impl<type, is_fp32_dest_acc_en, src_b_bcast_type, is_int_fpu_en, pack_mode>(
        DESC.shape.total_num_faces(), RegFmt, is_input_8bit_format);
}

template <
    ckernel::experimental::LLKMemDescriptor DESC,
    DataCopyType type = DataCopyType::A2D,
    bool is_fp32_dest_acc_en = false,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    bool unpack_to_dest = false>
inline void llk_math_eltwise_unary_datacopy(std::uint32_t dst_index) {
    constexpr std::uint8_t RegFmt = static_cast<std::uint8_t>(
        ckernel::infer_unpack_dst_format(static_cast<DataFormat>(DESC.format), is_fp32_dest_acc_en));
    llk_math_eltwise_unary_datacopy_impl<type, is_fp32_dest_acc_en, src_b_bcast_type, unpack_to_dest>(
        dst_index, DESC.format, RegFmt);
}
