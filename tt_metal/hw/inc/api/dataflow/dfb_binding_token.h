// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/llk_operand_members.h"

struct DFBBindingToken;

namespace ckernel {
namespace experimental {
struct LLKMemDescriptor;
constexpr LLKMemDescriptor to_llk_mem_descriptor(DFBBindingToken);
}  // namespace experimental
}  // namespace ckernel

// Opaque handle for a DataflowBuffer binding (declared in kernel_bindings_generated.h).
// The user will never directly interact with this type.
//
// The user's host code declares an accessor_name when binding a DFB endpoint to a kernel.
// The user then uses that accessor_name to construct a DataflowBuffer in the kernel code.
//
// Usage example:
//   // (Host code declares "my_dfb_name" as the DFB accessor name for this kernel.)
//   // In the kernel code:
//   DataflowBuffer my_dfb(dfb::my_dfb_name);
//
// Here my_dfb_name is a constexpr DFBBindingToken, auto-included in kernel_bindings_generated.h.
//
struct DFBBindingToken {
    explicit constexpr DFBBindingToken(uint16_t id) noexcept : id_(id) {}
    constexpr DFBBindingToken(uint16_t id, LlkOperandMembers llk) noexcept : id_(id), llk_(llk) {}

    // DFBBindingToken is backed by a compile-time ID (an implicit CTA).

    // Implicit conversion to uint32_t:
    // This lets a Metal 2.0 kernel pass a DFBBindingToken directly to Gen1 (WH/BH) LLK
    // compute APIs that expect a raw CB id.
    // This conversion is constexpr; it's intended for Gen1 use only.
    constexpr operator uint32_t() const noexcept { return id_; }

private:
    friend constexpr ckernel::experimental::LLKMemDescriptor ckernel::experimental::to_llk_mem_descriptor(
        DFBBindingToken);

    uint16_t id_;
    LlkOperandMembers llk_{};
};
