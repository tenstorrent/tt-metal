// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dfb_binding_token.h"
#include "api/scratchpad_binding_token.h"
#include "api/tensor/tensor_binding_token.h"

namespace ckernel {
namespace experimental {

constexpr LLKMemDescriptor llk_desc_from_members(LlkOperandMembers m) {
    return LLKMemDescriptor{m.format, TensorShape{m.face_r_dim, m.face_c_dim, m.num_faces_r_dim, m.num_faces_c_dim}};
}

constexpr LLKMemDescriptor to_llk_mem_descriptor(DFBBindingToken token) { return llk_desc_from_members(token.llk_); }

constexpr LLKMemDescriptor to_llk_mem_descriptor(ScratchpadBindingToken token) {
    return llk_desc_from_members(token.llk_);
}

template <uint32_t CTA, uint32_t CRTA>
constexpr LLKMemDescriptor to_llk_mem_descriptor(tensor_accessor::TensorBindingToken<CTA, CRTA> token) {
    static_assert(
        !tensor_accessor::TensorBindingToken<CTA, CRTA>::args_t::is_dram,
        "to_llk_mem_descriptor: DRAM TensorBindingToken has no node-local L1 region");
    return llk_desc_from_members(token.llk_);
}

}  // namespace experimental
}  // namespace ckernel
