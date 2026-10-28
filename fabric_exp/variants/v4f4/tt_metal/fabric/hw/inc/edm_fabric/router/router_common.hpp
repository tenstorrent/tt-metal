// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Compile-time facts both router programs use: stream register ids, the downstream direction maps, channel slot
// layouts and small register helpers. Everything here is derived from the compile-time args
// (fabric_erisc_router_ct_args.hpp); nothing here holds state.

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace fabric_router {

using tt::tt_fabric::eth_chan_directions;

// ---------------------------------------------------------------------------------------------------------------------
// Architecture and small constants
// ---------------------------------------------------------------------------------------------------------------------

constexpr bool is_blackhole_arch =
#if defined(ARCH_BLACKHOLE)
    true;
#else
    false;
#endif

constexpr bool is_2d = is_2d_fabric != 0;

// The count in a stream credit register (packets received, free slots). A count never exceeds its channel's slot
// count, and every channel static_asserts its slot count against this mask, so the low 11 bits hold it.
constexpr uint32_t stream_count_mask = 0x7FFu;
static_assert(stream_count_mask <= ((1u << REMOTE_DEST_WORDS_FREE_WIDTH) - 1));

// The Ethernet firmware's heartbeat word. While the router runs instead of the base firmware, each loop keeps it moving
// every 64 rounds, with UMD's fabric signature (0xDCBA) in the upper half.
#if defined(ARCH_BLACKHOLE)
constexpr uint32_t heartbeat_addr = 0x7CC70;
#else
constexpr uint32_t heartbeat_addr = 0x1F80;
#endif

// ---------------------------------------------------------------------------------------------------------------------
// Stream registers
// ---------------------------------------------------------------------------------------------------------------------

// The free-slots stream register of each sender channel, by fabric position (the table is fabric-flat: a router's
// compact channel i sits at fabric_position_for_compact_sender(i)).
constexpr std::array<uint32_t, MAX_NUM_SENDER_CHANNELS> sender_free_slots_streams = {
    sender_channel_0_free_slots_stream_id,
    sender_channel_1_free_slots_stream_id,
    sender_channel_2_free_slots_stream_id,
    sender_channel_3_free_slots_stream_id,
    sender_channel_4_free_slots_stream_id,
    sender_channel_5_free_slots_stream_id,
    sender_channel_6_free_slots_stream_id,
    sender_channel_7_free_slots_stream_id,
    sender_channel_8_free_slots_stream_id,
    sender_channel_9_free_slots_stream_id};

// The free-slots stream register of this router's compact sender channel CH.
template <size_t CH>
constexpr uint32_t sender_free_slots_stream = sender_free_slots_streams[fabric_position_for_compact_sender(CH)];

// Initialize a stream register that this router uses; k_unused_stream_id marks one it does not (a VC whose credits
// travel through L1 counters), and is out of register range.
template <uint32_t STREAM_ID>
FORCE_INLINE void init_stream_if_used(uint32_t value) {
    if constexpr (STREAM_ID != tt::tt_fabric::k_unused_stream_id) {
        init_ptr_val<STREAM_ID>(value);
    }
}

// The count held in stream register STREAM_ID (masked).
template <uint32_t STREAM_ID>
FORCE_INLINE uint32_t read_stream_count() {
    return NOC_STREAM_READ_REG(STREAM_ID, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX) & stream_count_mask;
}

// The raw value of stream register STREAM_ID (the caller masks).
template <uint32_t STREAM_ID>
FORCE_INLINE uint32_t read_stream_raw() {
    return NOC_STREAM_READ_REG(STREAM_ID, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
}

// ---------------------------------------------------------------------------------------------------------------------
// Downstream routers (2D): the other directions of this chip, in compact order
// ---------------------------------------------------------------------------------------------------------------------
//
// A router never forwards back out its own facing, so its downstream routers are the other four directions, packed
// densely in enum order (E, W, N, S, Z) with its own direction removed: the compact index of a downstream direction.

constexpr uint32_t max_downstreams = 4;

constexpr eth_chan_directions downstream_direction_table[eth_chan_directions::COUNT][max_downstreams] = {
    {eth_chan_directions::WEST, eth_chan_directions::NORTH, eth_chan_directions::SOUTH, eth_chan_directions::Z},
    {eth_chan_directions::EAST, eth_chan_directions::NORTH, eth_chan_directions::SOUTH, eth_chan_directions::Z},
    {eth_chan_directions::EAST, eth_chan_directions::WEST, eth_chan_directions::SOUTH, eth_chan_directions::Z},
    {eth_chan_directions::EAST, eth_chan_directions::WEST, eth_chan_directions::NORTH, eth_chan_directions::Z},
    {eth_chan_directions::EAST, eth_chan_directions::WEST, eth_chan_directions::NORTH, eth_chan_directions::SOUTH},
};

constexpr eth_chan_directions downstream_direction(size_t compact) {
    return downstream_direction_table[my_direction][compact];
}

// The compact index of downstream direction `dir` (dir != my_direction).
constexpr size_t downstream_compact_index(eth_chan_directions dir) {
    return static_cast<size_t>(dir) < static_cast<size_t>(my_direction) ? static_cast<size_t>(dir)
                                                                         : static_cast<size_t>(dir) - 1;
}

// Routing names a downstream by its direction (2D: action bit d is direction d); 1D's one downstream is named 0. The
// downstream tables keep the host's compact order, and downstream_entry maps a name to its entry there.
constexpr size_t downstream_entry(size_t name) {
    return is_2d ? downstream_compact_index(static_cast<eth_chan_directions>(name)) : 0;
}

// The free-slots stream register, on the downstream router, of the sender channel this router feeds (the downstream's
// channel for traffic arriving from this router's direction).
constexpr uint32_t downstream_sender_free_slots_stream(size_t vc, size_t compact) {
    const size_t base = vc == 0 ? 0 : VC1_FABRIC_POSITION_START - 1;
    if constexpr (is_2d) {
        const bool below = my_direction > static_cast<uint32_t>(downstream_direction(compact));
        return sender_free_slots_streams[base + (below ? my_direction : my_direction + 1)];
    } else {
        return sender_channel_1_free_slots_stream_id;
    }
}

// This router's local copy of a downstream's free slots (the downstream credits it as it sends), per VC.
constexpr std::array<uint32_t, max_downstreams> vc0_downstream_credit_streams = {
    vc_0_free_slots_from_downstream_edge_1_stream_id,
    vc_0_free_slots_from_downstream_edge_2_stream_id,
    vc_0_free_slots_from_downstream_edge_3_stream_id,
    vc_0_free_slots_from_downstream_edge_4_stream_id};
constexpr std::array<uint32_t, max_downstreams> vc1_downstream_credit_streams = {
    vc_1_free_slots_from_downstream_edge_1_stream_id,
    vc_1_free_slots_from_downstream_edge_2_stream_id,
    vc_1_free_slots_from_downstream_edge_3_stream_id,
    vc_1_free_slots_from_downstream_edge_4_stream_id};

constexpr uint32_t local_downstream_credit_stream(size_t vc, size_t compact) {
    if constexpr (is_2d) {
        return vc == 0 ? vc0_downstream_credit_streams[compact] : vc1_downstream_credit_streams[compact];
    } else {
        return vc0_downstream_credit_streams[0];
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Packet header
// ---------------------------------------------------------------------------------------------------------------------

template <typename D>
tt::tt_fabric::PacketHeaderBase<D>* packet_header_base_of(tt::tt_fabric::PacketHeaderBase<D>*);
using PacketHeaderBase =
    std::remove_pointer_t<decltype(packet_header_base_of(static_cast<PACKET_HEADER_TYPE*>(nullptr)))>;

// Payload size, send type and source channel are one header word, read with one load.
struct HeaderWord {
    uint16_t payload_size_bytes;
    tt::tt_fabric::NocSendType noc_send_type;
    uint8_t src_ch_id;

    static FORCE_INLINE HeaderWord load(const volatile PACKET_HEADER_TYPE* header) {
        static_assert(offsetof(PacketHeaderBase, payload_size_bytes) == 40);
        static_assert(offsetof(PacketHeaderBase, noc_send_type) == 42);
        static_assert(offsetof(PacketHeaderBase, src_ch_id) == 43);
        const uint32_t word = *reinterpret_cast<const volatile uint32_t*>(
            reinterpret_cast<uintptr_t>(header) + offsetof(PacketHeaderBase, payload_size_bytes));
        return {
            static_cast<uint16_t>(word & 0xFFFFu),
            static_cast<tt::tt_fabric::NocSendType>((word >> 16) & 0xFFu),
            static_cast<uint8_t>(word >> 24)};
    }
};

// ---------------------------------------------------------------------------------------------------------------------
// Compile-time loops
// ---------------------------------------------------------------------------------------------------------------------

// Calls f(std::integral_constant<size_t, I>) for each I in [0, N).
template <size_t N, typename F>
FORCE_INLINE void for_each_index(F&& f) {
    [&]<size_t... Is>(std::index_sequence<Is...>) { (f(std::integral_constant<size_t, Is>{}), ...); }(
        std::make_index_sequence<N>{});
}

// Calls f(std::integral_constant<size_t, I>) for each bit I set in MASK, in increasing order.
template <uint32_t MASK, typename F>
FORCE_INLINE void for_each_bit(F&& f) {
    for_each_index<32>([&](auto i) __attribute__((always_inline)) {
        if constexpr (((MASK >> decltype(i)::value) & 1u) != 0) {
            f(i);
        }
    });
}

constexpr uint32_t popcount(uint32_t mask) { return static_cast<uint32_t>(__builtin_popcount(mask)); }

// For plain recursion over a channel mask: the lowest channel in MASK, and MASK without it. The sender's per-pass code
// recurses instead of calling for_each_bit with a closure (router_sender.hpp, "Per-pass code takes no closures").
constexpr size_t lowest_bit(uint32_t mask) { return static_cast<size_t>(__builtin_ctz(mask)); }
constexpr uint32_t without_lowest_bit(uint32_t mask) { return mask & (mask - 1); }

}  // namespace fabric_router
