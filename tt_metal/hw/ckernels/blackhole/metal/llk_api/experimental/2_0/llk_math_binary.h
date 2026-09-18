// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "llk_math_binary_api.h"  // legacy CB-id API + unified llk_math_eltwise_binary_{init_,}impl
#include "api/compute/experimental/2_0/internal/llk_descriptor.h"

/*************************************************************************
 * LLK ELTWISE BINARY -- LLKOperand (id-free, compile-time NTTP) overloads
 *
 * Same function names as the CB-id API (llk_math_eltwise_binary_init / llk_math_eltwise_binary),
 * distinguished by taking an LLKMemDescriptor (of operand A -- the shape source, mirroring the CB-id API)
 * as the first NTTP. Eltwise binary math is FORMAT-FREE: it consumes only the tile geometry (DESC_A.shape);
 * register formats live in the unpacker/dest config from compute_kernel_hw_startup. Both overloads forward
 * to the same unified cores as the CB-id API.
 *************************************************************************/

template <
    ckernel::experimental::LLKMemDescriptor DESC_A,
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type = BroadcastType::NONE,
    MathFidelity math_fidelity = MathFidelity::LoFi,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_eltwise_binary_init(const std::uint32_t acc_to_dest = 0) {
    llk_math_eltwise_binary_init_impl<eltwise_binary_type, src_b_bcast_type, math_fidelity, binary_reuse_dest>(
        DESC_A.shape, acc_to_dest);
}

template <
    ckernel::experimental::LLKMemDescriptor DESC_A,
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_eltwise_binary(std::uint32_t dst_index, const bool clear_fp32_dst_acc = true) {
    llk_math_eltwise_binary_impl<
        eltwise_binary_type,
        src_b_bcast_type,
        is_fp32_dest_acc_en,
        math_fidelity,
        binary_reuse_dest>(DESC_A.shape, dst_index, clear_fp32_dst_acc);
}
