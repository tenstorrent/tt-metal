// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <cstdint>
#include <tuple>
#include <utility>
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp"
#include "risc_attribs.h"

// taken from fabric_erisc_datamover.cpp ... commonize!
// Forward‐declare the Impl primary template:
template <template <uint8_t> class ChannelType, auto& BufferSizes, typename Seq>
struct ChannelPointersTupleImpl;

// Provide the specialization that actually holds the tuple and `get<>`:
template <template <uint8_t> class ChannelType, auto& BufferSizes, size_t... Is>
struct ChannelPointersTupleImpl<ChannelType, BufferSizes, std::index_sequence<Is...>> {
    std::tuple<ChannelType<BufferSizes[Is]>...> channel_ptrs;

    template <size_t I>
    constexpr auto& get() {
        return std::get<I>(channel_ptrs);
    }
};

// Simplify the “builder” so that make() returns the Impl<…> directly:
template <template <uint8_t> class ChannelType, auto& BufferSizes>
struct ChannelPointersTuple {
    static constexpr size_t N = std::size(BufferSizes);

    static constexpr auto make() {
        return ChannelPointersTupleImpl<ChannelType, BufferSizes, std::make_index_sequence<N>>{};
    }
};

/*
 * Tracks receiver channel pointers (from sender side)
 */
template <uint8_t RECEIVER_NUM_BUFFERS>
struct OutboundReceiverChannelPointers {
    uint32_t num_free_slots = RECEIVER_NUM_BUFFERS;
    tt::tt_fabric::BufferIndex remote_receiver_buffer_index{0};
    size_t cached_next_buffer_slot_addr = 0;

    FORCE_INLINE bool has_space_for_packet() const { return num_free_slots; }
};

/*
 * Tracks receiver channel pointers (from receiver side). Must call reset() before using.
 */
template <uint8_t RECEIVER_NUM_BUFFERS>
struct ReceiverChannelPointers {
    tt::tt_fabric::ChannelCounter<RECEIVER_NUM_BUFFERS> wr_sent_counter;
    tt::tt_fabric::ChannelCounter<RECEIVER_NUM_BUFFERS> wr_flush_counter;
    tt::tt_fabric::ChannelCounter<RECEIVER_NUM_BUFFERS> ack_counter;
    tt::tt_fabric::ChannelCounter<RECEIVER_NUM_BUFFERS> completion_counter;
    std::array<uint8_t, RECEIVER_NUM_BUFFERS> src_chan_ids;

    FORCE_INLINE void set_src_chan_id(tt::tt_fabric::BufferIndex buffer_index, uint8_t src_chan_id) {
        src_chan_ids[buffer_index.get()] = src_chan_id;
    }

    FORCE_INLINE uint8_t get_src_chan_id(tt::tt_fabric::BufferIndex buffer_index) const {
        return src_chan_ids[buffer_index.get()];
    }

    FORCE_INLINE void reset() {
        wr_sent_counter.reset();
        wr_flush_counter.reset();
        ack_counter.reset();
        completion_counter.reset();
    }
};
