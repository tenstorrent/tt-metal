// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "llk_unpack_AB_api.h"  // legacy CB-id API + unified llk_unpack_AB_init_impl / llk_unpack_AB_impl
#include "api/compute/experimental/2_0/llk_mem_descriptor.h"

/*************************************************************************
 * LLK UNPACK AB -- LLKOperand (id-free, compile-time NTTP) overloads
 *
 * Same function names as the CB-id API (llk_unpack_AB_init / llk_unpack_AB), distinguished by taking an
 * LLKMemDescriptor (of operand A -- the shape source, mirroring the CB-id API) as the first NTTP. AB unpack
 * is FORMAT-FREE at the op level (src/dst formats are programmed once at compute_kernel_hw_startup), so the
 * op forwards only the two runtime L1 base pointers (from llk_mem_descriptor.h::cb_read_address). Both
 * overloads forward to the same unified cores as the CB-id API. ROW-broadcast is not part of the id-free
 * surface yet (would need operand B's L1 format); only BroadcastType::NONE is exercised.
 *************************************************************************/

template <ckernel::experimental::LLKMemDescriptor DESC_A, BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB_init(const ckernel::Transpose transpose = ckernel::Transpose::None) {
    static_assert(BType != BroadcastType::ROW, "id-free llk_unpack_AB: ROW broadcast not supported yet");
    llk_unpack_AB_init_impl<BType>(DESC_A.shape, transpose);
}

template <ckernel::experimental::LLKMemDescriptor DESC_A, BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB(std::uint32_t base_ptr_a, std::uint32_t base_ptr_b) {
    static_assert(BType != BroadcastType::ROW, "id-free llk_unpack_AB: ROW broadcast not supported yet");
    llk_unpack_AB_impl<BType>(base_ptr_a, base_ptr_b, 0 /*bcast_row_idx*/, 0 /*operandB_src_format*/);
}
