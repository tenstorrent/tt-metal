// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/core_local_mem.h"
#include "api/debug/assert.h"

// Binding token for a Program-scope scratchpad, emitted into the `scratch::` namespace of
// kernel_bindings_generated.h. The per-node base L1 address is injected at enqueue time via a CRTA
// slot (crta_offset is its word index in the common-runtime-args buffer); the size is static (known
// from ScratchpadSpec at spec time) and baked in here. Mirrors TensorAccessorBindingToken's
// static-CTA / runtime-CRTA split, address-only.
struct ScratchpadAccessor {
    uint32_t crta_offset;    // word index of the base-address slot in the CRTA buffer
    uint32_t size_in_bytes;  // static per-node size
};

template <typename T>
class Scratchpad {
public:
    // Resolve a scratchpad from its binding token: read the per-node base L1 address from the CRTA
    // slot at accessor.crta_offset; size_in_bytes is the static spec value. Not constexpr — it reads
    // a runtime CRTA value.
    [[nodiscard]] explicit Scratchpad(const ScratchpadAccessor& accessor) noexcept :
        Scratchpad(
            CoreLocalMem<T>{get_common_arg_val<uint32_t>(static_cast<int>(accessor.crta_offset))},
            accessor.size_in_bytes) {}

    [[nodiscard]] constexpr Scratchpad(CoreLocalMem<T> base_addr, uint32_t size_in_bytes) noexcept :
        start_addr_(base_addr), end_addr_(CoreLocalMem<T>{base_addr.get_address() + size_in_bytes}) {
        ASSERT(base_addr.get_address() % alignof(T) == 0);
        ASSERT(size_in_bytes % sizeof(T) == 0);
    }

    // Value based const semantics

    /** @brief Get the element at the given index
     *
     * @param index The index of the element to get
     * @return Reference to the element at the given index
     */
    [[nodiscard]] T& operator[](uint32_t index) {
        auto location = start_addr_ + index;
        ASSERT(location < end_addr_);
        return *location;
    }

    /** @brief Get the element at the given index
     *
     * @param index The index of the element to get
     * @return Reference to the element at the given index
     */
    [[nodiscard]] const T& operator[](uint32_t index) const {
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
