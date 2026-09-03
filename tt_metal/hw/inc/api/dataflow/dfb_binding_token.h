// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

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
// This header holds only the token, with no dependency beyond <cstdint>, so the generated
// bindings header (and anything else that just needs to name a binding) does not have to pull
// in the whole DataflowBuffer implementation. See api/dataflow/dataflow_buffer.h for the
// DataflowBuffer class this token constructs.
//
struct DFBBindingToken {
    explicit constexpr DFBBindingToken(uint16_t id) noexcept : id_(id) {}

    // DFBBindingToken is backed by a compile-time ID (an implicit CTA).

    // Implicit conversion to uint32_t:
    // This lets a Metal 2.0 kernel pass a DFBBindingToken directly to Gen1 (WH/BH) LLK
    // compute APIs that expect a raw CB id.
    // This conversion is constexpr; it's intended for Gen1 use only.
    constexpr operator uint32_t() const noexcept { return id_; }

private:
    uint16_t id_;
};
