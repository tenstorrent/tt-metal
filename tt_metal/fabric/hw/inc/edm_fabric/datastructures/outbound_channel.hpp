// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "risc_attribs.h"
#include <cstdint>
#include <cstddef>

namespace tt::tt_fabric {

// A base sender channel interface class that will be specialized for different
// channel architectures (e.g. static vs elastic sizing)
template <typename HEADER_TYPE, uint8_t NUM_BUFFERS, typename DERIVED_T>
class SenderEthChannelInterface {
public:
    explicit SenderEthChannelInterface() = default;

    FORCE_INLINE void init(
        size_t channel_base_address, size_t max_eth_payload_size_in_bytes, size_t header_size_bytes) {
        static_cast<DERIVED_T*>(this)->init_impl(
            channel_base_address, max_eth_payload_size_in_bytes, header_size_bytes);
    }

    FORCE_INLINE size_t get_cached_next_buffer_slot_addr() const {
        return static_cast<const DERIVED_T*>(this)->get_cached_next_buffer_slot_addr_impl();
    }

    FORCE_INLINE void advance_to_next_cached_buffer_slot_addr() {
        static_cast<DERIVED_T*>(this)->advance_to_next_cached_buffer_slot_addr_impl();
    }
};

}  // namespace tt::tt_fabric
