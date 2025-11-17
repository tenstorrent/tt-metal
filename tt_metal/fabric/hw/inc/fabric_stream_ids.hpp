// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <array>

namespace tt::tt_fabric {

/**
 * @brief Stream register assignments for fabric datamover
 *
 * This struct defines the stream IDs used by the fabric datamover for various purposes.
 * Stream IDs 0-26 are used by the fabric datamover.
 * Stream IDs 27-32 are reserved for lite fabric to avoid conflicts.
 *
 * @note When modifying these assignments, ensure no conflicts with lite_fabric/hw/inc/constants.hpp
 */
struct StreamRegAssignments {
    // Packet send/complete stream IDs
    static constexpr uint32_t to_receiver_0_pkts_sent_id = 0;
    static constexpr uint32_t to_receiver_1_pkts_sent_id = 1;
    // Packet acknowledgment stream IDs (only sender channels 0-3)
    static constexpr uint32_t to_sender_0_pkts_acked_id = 2;
    static constexpr uint32_t to_sender_1_pkts_acked_id = 3;
    static constexpr uint32_t to_sender_2_pkts_acked_id = 4;
    static constexpr uint32_t to_sender_3_pkts_acked_id = 5;
    // Packet completion stream IDs (sender channels 0-6)
    static constexpr uint32_t to_sender_0_pkts_completed_id = 6;
    static constexpr uint32_t to_sender_1_pkts_completed_id = 7;
    static constexpr uint32_t to_sender_2_pkts_completed_id = 8;
    static constexpr uint32_t to_sender_3_pkts_completed_id = 9;
    static constexpr uint32_t to_sender_4_pkts_completed_id = 10;  // VC1 compact 0 (2D only)
    static constexpr uint32_t to_sender_5_pkts_completed_id = 11;  // VC1 compact 1 (2D only)
    static constexpr uint32_t to_sender_6_pkts_completed_id = 12;  // VC1 compact 2 (2D only)
    // Receiver channel free slots stream IDs
    static constexpr uint32_t vc_0_free_slots_from_downstream_edge_1 = 13;
    static constexpr uint32_t vc_0_free_slots_from_downstream_edge_2 = 14;
    static constexpr uint32_t vc_0_free_slots_from_downstream_edge_3 = 15;
    static constexpr uint32_t vc_1_free_slots_from_downstream_edge_1 = 16;  // VC1 (2D only)
    static constexpr uint32_t vc_1_free_slots_from_downstream_edge_2 = 17;  // VC1 (2D only)
    static constexpr uint32_t vc_1_free_slots_from_downstream_edge_3 = 18;  // VC1 (2D only)
    // Sender channel free slots stream IDs
    static constexpr uint32_t sender_channel_0_free_slots_stream_id = 19;  // for tensix worker
    static constexpr uint32_t sender_channel_1_free_slots_stream_id = 20;  // VC0 compact 0
    static constexpr uint32_t sender_channel_2_free_slots_stream_id = 21;  // VC0 compact 1
    static constexpr uint32_t sender_channel_3_free_slots_stream_id = 22;  // VC0 compact 2
    static constexpr uint32_t sender_channel_4_free_slots_stream_id = 23;  // VC1 compact 0 (2D only)
    static constexpr uint32_t sender_channel_5_free_slots_stream_id = 24;  // VC1 compact 1 (2D only)
    static constexpr uint32_t sender_channel_6_free_slots_stream_id = 25;  // VC1 compact 2 (2D only)
    // Local tensix relay free slots stream ID (UDM mode only)
    static constexpr uint32_t tensix_relay_local_free_slots_stream_id = 26;
    // Used by Lite Fabric (stream IDs 27-32 reserved)
    // Consult tt_metal/lite_fabric/hw/inc/constants.hpp to ensure no conflicts
    static constexpr uint32_t reserved_lite_fabric_0_stream_id = 26;
    static constexpr uint32_t reserved_lite_fabric_1_stream_id = 27;
    static constexpr uint32_t reserved_lite_fabric_2_stream_id = 28;
    static constexpr uint32_t reserved_lite_fabric_3_stream_id = 29;
    static constexpr uint32_t reserved_lite_fabric_4_stream_id = 30;
    static constexpr uint32_t reserved_lite_fabric_5_stream_id = 31;
    // Multi-RISC teardown synchronization stream ID (reuses to_receiver_0_pkts_sent_id)
    static constexpr uint32_t multi_risc_teardown_sync_stream_id = 0;

    static const auto& get_all_stream_ids() {
        static constexpr std::array stream_ids = {
            to_receiver_0_pkts_sent_id,
            to_receiver_1_pkts_sent_id,
            to_sender_0_pkts_acked_id,
            to_sender_1_pkts_acked_id,
            to_sender_2_pkts_acked_id,
            to_sender_3_pkts_acked_id,
            to_sender_0_pkts_completed_id,
            to_sender_1_pkts_completed_id,
            to_sender_2_pkts_completed_id,
            to_sender_3_pkts_completed_id,
            to_sender_4_pkts_completed_id,
            to_sender_5_pkts_completed_id,
            to_sender_6_pkts_completed_id,
            vc_0_free_slots_from_downstream_edge_1,
            vc_0_free_slots_from_downstream_edge_2,
            vc_0_free_slots_from_downstream_edge_3,
            vc_1_free_slots_from_downstream_edge_1,
            vc_1_free_slots_from_downstream_edge_2,
            vc_1_free_slots_from_downstream_edge_3,
            sender_channel_0_free_slots_stream_id,
            sender_channel_1_free_slots_stream_id,
            sender_channel_2_free_slots_stream_id,
            sender_channel_3_free_slots_stream_id,
            sender_channel_4_free_slots_stream_id,
            sender_channel_5_free_slots_stream_id,
            sender_channel_6_free_slots_stream_id,
            tensix_relay_local_free_slots_stream_id,
            multi_risc_teardown_sync_stream_id};
        return stream_ids;  // 28 stream IDs
    }
};

}  // namespace tt::tt_fabric
