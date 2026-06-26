// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/core_local_mem.h"
#include "api/debug/assert.h"

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
// Here my_scratchpad_name is a constexpr ScratchpadAccessor, auto-included in
// kernel_bindings_generated.h.
class ScratchpadAccessor {
public:
    explicit constexpr ScratchpadAccessor(uint32_t crta_offset, uint32_t size_in_bytes) noexcept :
        crta_offset_(crta_offset), size_in_bytes_(size_in_bytes) {}

private:
    template <typename T>
    friend class Scratchpad;

    uint32_t crta_offset_;    // word index of the base-address slot in the CRTA buffer
    uint32_t size_in_bytes_;  // static per-node size
};

template <typename T>
class Scratchpad {
public:
    // Resolve a scratchpad from its binding token: read the per-node base L1 address from the CRTA
    // slot at the token's crta offset; size is the static spec value.
    [[nodiscard]] explicit Scratchpad(const ScratchpadAccessor& accessor) noexcept :
        Scratchpad(
            CoreLocalMem<T>{get_common_arg_val<uint32_t>(static_cast<int>(accessor.crta_offset_))},
            accessor.size_in_bytes_) {}

    [[nodiscard]] constexpr Scratchpad(CoreLocalMem<T> base_addr, uint32_t size_in_bytes) noexcept :
        start_addr_(base_addr), end_addr_(CoreLocalMem<T>{base_addr.get_address() + size_in_bytes}) {
        ASSERT(base_addr.get_address() % alignof(T) == 0);
        ASSERT(size_in_bytes % sizeof(T) == 0);
    }

    /** @brief Get the element at the given index
     *
     * @param index The index of the element to get
     * @return Reference to the element at the given index
     */
    [[nodiscard]] T& operator[](uint32_t index) const {
        auto location = start_addr_ + index;
        ASSERT(location < end_addr_);
        return *location;
    }

    [[nodiscard]] constexpr uint32_t size() const noexcept { return size_in_bytes() / sizeof(T); }

    [[nodiscard]] constexpr uint32_t size_in_bytes() const noexcept {
        return end_addr_.get_address() - start_addr_.get_address();
    }

    [[nodiscard]] constexpr CoreLocalMem<T> get_base_addr() const noexcept { return start_addr_; }

private:
    // Invariant:
    // - start_addr_ and end_addr_ is always aligned to the alignment of T.
    // - `end_addr_` - `start_addr_` should always be a multiple of sizeof(T).
    CoreLocalMem<T> start_addr_, end_addr_;
};
