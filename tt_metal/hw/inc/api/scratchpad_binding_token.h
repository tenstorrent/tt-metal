// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/llk_operand_members.h"

class ScratchpadBindingToken;

namespace ckernel {
namespace experimental {
struct LLKMemDescriptor;
constexpr LLKMemDescriptor to_llk_mem_descriptor(ScratchpadBindingToken);
}  // namespace experimental
}  // namespace ckernel

template <typename T>
class Scratchpad;

// Opaque handle for a Program-scope scratchpad binding (declared in kernel_bindings_generated.h).
// The user will never directly interact with this type.
//
// The user's host code declares an accessor_name when binding a scratchpad to a kernel.
// The user then uses that accessor_name to construct a Scratchpad in the kernel code.
//
// Usage example:
//   // (Host code declares "my_scratchpad_name" as the scratchpad accessor name for this kernel.)
//   // In the kernel code:
//   Scratchpad<int32_t> my_pad(scratch::my_scratchpad_name);
//
// Here my_scratchpad_name is a constexpr ScratchpadBindingToken, auto-included in
// kernel_bindings_generated.h.
class ScratchpadBindingToken {
public:
    explicit constexpr ScratchpadBindingToken(uint32_t crta_offset, uint32_t size_in_bytes) noexcept :
        crta_offset_(crta_offset), size_in_bytes_(size_in_bytes) {}
    constexpr ScratchpadBindingToken(uint32_t crta_offset, uint32_t size_in_bytes, LlkOperandMembers llk) noexcept :
        crta_offset_(crta_offset), size_in_bytes_(size_in_bytes), llk_(llk) {}

private:
    template <typename T>
    friend class Scratchpad;
    friend constexpr ckernel::experimental::LLKMemDescriptor ckernel::experimental::to_llk_mem_descriptor(
        ScratchpadBindingToken);

    uint32_t crta_offset_;    // word index of the base-address slot in the CRTA buffer
    uint32_t size_in_bytes_;  // static per-node size
    LlkOperandMembers llk_{};
};
