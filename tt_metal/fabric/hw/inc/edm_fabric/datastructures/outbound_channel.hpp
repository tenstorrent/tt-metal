// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "internal/risc_attribs.h"
#include <cstdint>
#include <cstddef>

namespace tt::tt_fabric {

// A base sender channel interface class
template <typename HEADER_TYPE, uint8_t NUM_BUFFERS, typename DERIVED_T>
class SenderEthChannelInterface {
public:
    explicit SenderEthChannelInterface() = default;

    // Const-addressed channels take no arguments; runtime-addressed ones take the base address,
    // buffer size and header size.
    template <typename... Args>
    FORCE_INLINE void init(Args... args) {
        static_cast<DERIVED_T*>(this)->init_impl(args...);
    }

    FORCE_INLINE size_t get_cached_next_buffer_slot_addr() const {
        return static_cast<const DERIVED_T*>(this)->get_cached_next_buffer_slot_addr_impl();
    }

    FORCE_INLINE void advance_to_next_cached_buffer_slot_addr() {
        static_cast<DERIVED_T*>(this)->advance_to_next_cached_buffer_slot_addr_impl();
    }
};

}  // namespace tt::tt_fabric
