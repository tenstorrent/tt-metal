// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "llk_unpack_A_api.h"  // legacy CB-id API + the unified llk_unpack_A_init_impl / llk_unpack_A_impl
#include "data_format_derive.h"
#include "api/compute/experimental/2_0/internal/llk_descriptor.h"

/*************************************************************************
 * LLK UNPACK A -- LLKOperand (id-free, compile-time NTTP) overloads
 *
 * Same function names as the CB-id API (llk_unpack_A_init / llk_unpack_A), distinguished by taking an
 * LLKMemDescriptor as the first NTTP instead of a CB id. DESC carries the buffer L1 format + geometry
 * (folds -> DCE); the SrcA register format is derived HERE from DESC.format + the fp32-dest-acc flag
 * (data_format_derive.h) and never exposed above the LLK. Both overloads forward to the same unified
 * cores as the CB-id API. The runtime base pointer comes from the caller via cb_operand_helpers.h::cb_read_address.
 *************************************************************************/

template <
    ckernel::experimental::LLKMemDescriptor DESC,
    bool is_fp32_dest_acc_en = false,
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A_init(
    const std::uint32_t transpose_of_faces = 0, const std::uint32_t within_face_16x16_transpose = 0) {
    constexpr std::uint8_t RegFmt = ckernel::infer_unpack_reg_fmt(DESC.format, is_fp32_dest_acc_en);
    llk_unpack_A_init_impl<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(
        transpose_of_faces, within_face_16x16_transpose, DESC.shape, static_cast<std::uint32_t>(DESC.format), RegFmt);
}

template <
    ckernel::experimental::LLKMemDescriptor DESC,
    bool is_fp32_dest_acc_en = false,
    BroadcastType BType = BroadcastType::NONE,
    bool acc_to_dest = false,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool unpack_to_dest = false>
inline void llk_unpack_A(std::uint32_t base_ptr) {
    constexpr std::uint8_t RegFmt = ckernel::infer_unpack_reg_fmt(DESC.format, is_fp32_dest_acc_en);
    llk_unpack_A_impl<BType, acc_to_dest, binary_reuse_dest, unpack_to_dest>(
        base_ptr, static_cast<std::uint32_t>(DESC.format), RegFmt);
}
