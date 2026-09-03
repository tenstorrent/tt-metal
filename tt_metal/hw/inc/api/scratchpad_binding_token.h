// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "internal/llk_metadata.h"

// Forward declaration only: the token grants Scratchpad<T> access to its members, but nothing
// in this header needs the definition. Defined in api/scratchpad.h.
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
//
// This header holds only the token, with no dependency beyond <cstdint> plus the LLK metadata
// sidecar, so the generated bindings header (and anything else that just needs to name a binding)
// does not have to pull in the whole Scratchpad implementation. See api/scratchpad.h for the
// Scratchpad class this token constructs.
namespace binding_details {
template <const auto& Token>
struct LLKOperandExtractor;
}

class ScratchpadBindingToken {
public:
    // Construct a Scratchpad from the offset and size supplied by the host.
    //
    // This Scratchpad does not contain any llk metadata.
    // Attempt to extract an LLKOperand from this token using LLKOperandFrom will cause a compile-time error.
    explicit constexpr ScratchpadBindingToken(uint32_t crta_offset, uint32_t size_in_bytes) noexcept :
        crta_offset_(crta_offset), size_in_bytes_(size_in_bytes) {}

    // Optional binding token constructor used when the host supplies LLK metadata.
    // See "Entry format metadata" in ScratchpadSpec.
    constexpr ScratchpadBindingToken(
        uint32_t crta_offset, uint32_t size_in_bytes, binding_details::LLKMetadata llk) noexcept :
        crta_offset_(crta_offset), size_in_bytes_(size_in_bytes), llk_metadata_(llk) {}

private:
    template <typename T>
    friend class Scratchpad;

    template <const auto& Token>
    friend struct binding_details::LLKOperandExtractor;

    uint32_t crta_offset_;    // word index of the base-address slot in the CRTA buffer
    uint32_t size_in_bytes_;  // static per-node size
    binding_details::LLKMetadata llk_metadata_{};
};
