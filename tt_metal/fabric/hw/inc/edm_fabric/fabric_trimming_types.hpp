// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Host+device compatible header: data layout and query accessors for datapath usage capture.
// No device-specific includes.
// Mutation helpers (set/update) are free functions in fabric_trimming.hpp (device-only).

#include <array>
#include <cstdint>
#include <cstddef>

namespace tt::tt_fabric {

// Convert a receiver router's compact 2D downstream-slot mask into the destination routers'
// flattened sender-channel IDs. Each compact slot's 4-bit channel ID is supplied by host wiring.
constexpr uint16_t forwarded_slots_to_sender_mask(
    uint8_t forwarded_slot_mask, uint32_t packed_downstream_sender_channel_ids) {
    constexpr uint32_t sender_channel_id_width_bits = 4;
    constexpr uint32_t sender_channel_id_mask = (1u << sender_channel_id_width_bits) - 1u;
    uint16_t sender_mask = 0;
    for (uint32_t compact_slot = 0; compact_slot < 4; ++compact_slot) {
        if ((forwarded_slot_mask & (1u << compact_slot)) != 0) {
            const uint32_t sender_channel_id =
                (packed_downstream_sender_channel_ids >> (compact_slot * sender_channel_id_width_bits)) &
                sender_channel_id_mask;
            sender_mask |= static_cast<uint16_t>(1u << sender_channel_id);
        }
    }
    return sender_mask;
}

// Primary template - enabled implementation (full data storage)
template <bool ENABLED, size_t NUM_VC = 2, size_t MAX_NUM_SENDER_CHANNELS = 9>
struct FabricDatapathUsageL1Results {
    using SenderChannelUsedBitfield = uint16_t;
    using ReceiverChannelDataForwardedBitfield = uint16_t;
    using NocSendTypeBitfield = uint16_t;

    // Record the max and min packet size seen by each sender channel
    std::array<uint16_t, MAX_NUM_SENDER_CHANNELS> sender_channel_min_packet_size_seen_bytes_by_vc = {};
    std::array<uint16_t, MAX_NUM_SENDER_CHANNELS> sender_channel_max_packet_size_seen_bytes_by_vc = {};

    // A bit is set high if the sender channel with ID matching that bit (offset) processed any traffic
    SenderChannelUsedBitfield sender_channel_used_bitfield_by_vc = {};

    // For each receiving VC, bits encode destination routers' flattened sender-channel IDs.
    // They indicate forwarding topology, not local sender use. A local-only delivery leaves the row unchanged.
    std::array<SenderChannelUsedBitfield, NUM_VC> sender_channel_forwarded_to_bitfield_by_vc = {};

    // A bit is set high if the receiver channel with ID matching that bit (offset) has forwarded any traffic
    ReceiverChannelDataForwardedBitfield receiver_channel_data_forwarded_bitfield_by_vc = {};

    // A bit is set high if the receiver channel on this VC forwards a noc message of that type
    std::array<NocSendTypeBitfield, NUM_VC> used_noc_send_type_by_vc_bitfield = {};

    // Query accessors
    bool is_sender_channel_used(size_t sender_channel_id) const {
        return (sender_channel_used_bitfield_by_vc & (1u << sender_channel_id)) != 0;
    }
    bool is_receiver_channel_data_forwarded(size_t receiver_channel_id) const {
        return (receiver_channel_data_forwarded_bitfield_by_vc & (1u << receiver_channel_id)) != 0;
    }

    bool operator==(const FabricDatapathUsageL1Results& other) const {
        return sender_channel_min_packet_size_seen_bytes_by_vc == other.sender_channel_min_packet_size_seen_bytes_by_vc &&
               sender_channel_max_packet_size_seen_bytes_by_vc == other.sender_channel_max_packet_size_seen_bytes_by_vc &&
               sender_channel_used_bitfield_by_vc == other.sender_channel_used_bitfield_by_vc &&
               sender_channel_forwarded_to_bitfield_by_vc == other.sender_channel_forwarded_to_bitfield_by_vc &&
               receiver_channel_data_forwarded_bitfield_by_vc == other.receiver_channel_data_forwarded_bitfield_by_vc &&
               used_noc_send_type_by_vc_bitfield == other.used_noc_send_type_by_vc_bitfield;
    }
    bool operator!=(const FabricDatapathUsageL1Results& other) const { return !(*this == other); }
};

// Specialization for disabled implementation - zero overhead, no storage
template <size_t NUM_VC, size_t MAX_NUM_SENDER_CHANNELS>
struct FabricDatapathUsageL1Results<false, NUM_VC, MAX_NUM_SENDER_CHANNELS> {};

}  // namespace tt::tt_fabric
