// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/llk_operand_members.h"

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
    uint16_t id_;
    LlkOperandMembers llk_{};
};

// Compile-time handle for a CrossNode/PrefetcherPipe *relay* local DFB binding.
// Distinct from DFBBindingToken so kernels cannot silently treat a normal DFB as a
// relay (or vice versa) without a cast — no runtime "am I a relay?" check needed.
// Emitted into kernel_bindings_generated.h when the host DFB was created as a relay.
//
// PrefetcherPipe relays additionally carry the prefetcher_pipe_id so the TRISC-side
// DataflowBuffer constructor can O(1)-index that slot in the launch-msg persistent
// region and snap the borrowed local iface to the durable fifo_ptr checkpoint.
// CrossNode relays omit it (NO_PREFETCHER_PIPE): CrossNode state is re-zeroed every
// launch, so the dispatch-written local CB config is already correct.
struct RelayDFBBindingToken {
    static constexpr uint8_t NO_PREFETCHER_PIPE = 0xFF;

    explicit constexpr RelayDFBBindingToken(uint16_t id, uint8_t prefetcher_pipe_id = NO_PREFETCHER_PIPE) noexcept :
        id_(id), prefetcher_pipe_id_(prefetcher_pipe_id) {}
    constexpr RelayDFBBindingToken(uint16_t id, uint8_t prefetcher_pipe_id, LlkOperandMembers llk) noexcept :
        id_(id), prefetcher_pipe_id_(prefetcher_pipe_id), llk_(llk) {}

    constexpr operator uint32_t() const noexcept { return id_; }

    constexpr uint8_t prefetcher_pipe_id() const noexcept { return prefetcher_pipe_id_; }

private:
    uint16_t id_;
    uint8_t prefetcher_pipe_id_;
    LlkOperandMembers llk_{};
};
