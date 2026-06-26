// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/core_local_mem.h"
#include "api/debug/assert.h"

// TODO: Write doc
struct ScratchpadAccessor {
    uint16_t id;
    // entry size could be injected.
};

template <typename T>
class Scratchpad {
public:
    // TODO: Implement this.
    [[nodiscard]] explicit Scratchpad(const ScratchpadAccessor& accessor) noexcept;

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
