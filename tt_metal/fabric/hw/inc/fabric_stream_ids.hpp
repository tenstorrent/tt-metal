// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <array>

namespace tt::tt_fabric {

/*
 * Stream Register Assignments for Fabric EDM
 *
 * These stream IDs are used by both fabric EDM (erisc_datamover_builder) and lite_fabric.
 * This header can be included in both host and device code (kernel builds).
 *
 * IMPORTANT: When modifying stream IDs, ensure:
 * - No conflicts between fabric EDM and lite_fabric usage
 * - Consult tt_metal/lite_fabric/hw/inc/constants.hpp for lite_fabric dependencies
 */
struct StreamRegAssignments {
    // Packet send/complete stream IDs
    static constexpr uint32_t to_receiver_0_pkts_sent_id = 0;
    static constexpr uint32_t to_receiver_1_pkts_sent_id = 1;
    static constexpr uint32_t to_receiver_2_pkts_sent_id = 2;  // VC2 (2D only)
    // to_sender_X_pkts_acked_id removed - ack streams no longer used
    static constexpr uint32_t to_sender_0_pkts_completed_id = 3;
    static constexpr uint32_t to_sender_1_pkts_completed_id = 4;
    static constexpr uint32_t to_sender_2_pkts_completed_id = 5;
    static constexpr uint32_t to_sender_3_pkts_completed_id = 6;
    static constexpr uint32_t to_sender_4_pkts_completed_id = 7;   // VC1
    static constexpr uint32_t to_sender_5_pkts_completed_id = 8;   // VC2 sender ch 0 (2D only)
    static constexpr uint32_t to_sender_6_pkts_completed_id = 9;   // VC2 sender ch 1 (2D only)
    static constexpr uint32_t to_sender_7_pkts_completed_id = 10;  // VC2 sender ch 2 (2D only)
    // Receiver channel free slots stream IDs
    static constexpr uint32_t vc_0_free_slots_from_downstream_edge_1 = 11;
    static constexpr uint32_t vc_0_free_slots_from_downstream_edge_2 = 12;
    static constexpr uint32_t vc_0_free_slots_from_downstream_edge_3 = 13;
    static constexpr uint32_t vc_1_free_slots_from_downstream_edge_1 = 14;
    static constexpr uint32_t vc_2_free_slots_from_downstream_edge_1 = 15;  // VC2 (2D only)
    static constexpr uint32_t vc_2_free_slots_from_downstream_edge_2 = 16;  // VC2 (2D only)
    static constexpr uint32_t vc_2_free_slots_from_downstream_edge_3 = 17;  // VC2 (2D only)
    // Sender channel free slots stream IDs
    static constexpr uint32_t sender_channel_0_free_slots_stream_id = 18;  // for tensix worker
    static constexpr uint32_t sender_channel_1_free_slots_stream_id = 19;  // VC0 compact 0
    static constexpr uint32_t sender_channel_2_free_slots_stream_id = 20;  // VC0 compact 1
    static constexpr uint32_t sender_channel_3_free_slots_stream_id = 21;  // VC0 compact 2
    static constexpr uint32_t sender_channel_4_free_slots_stream_id = 22;  // VC1
    static constexpr uint32_t sender_channel_5_free_slots_stream_id = 23;  // VC2 compact 0 (2D only)
    static constexpr uint32_t sender_channel_6_free_slots_stream_id = 24;  // VC2 compact 1 (2D only)
    static constexpr uint32_t sender_channel_7_free_slots_stream_id = 25;  // VC2 compact 2 (2D only)
    // Used by Lite Fabric
    // Consult tt_metal/lite_fabric/hw/inc/constants.hpp to ensure no conflicts
    static constexpr uint32_t reserved_lite_fabric_0_stream_id = 26;
    static constexpr uint32_t reserved_lite_fabric_1_stream_id = 27;
    static constexpr uint32_t reserved_lite_fabric_2_stream_id = 28;
    static constexpr uint32_t reserved_lite_fabric_3_stream_id = 29;
    static constexpr uint32_t reserved_lite_fabric_4_stream_id = 30;
    static constexpr uint32_t reserved_lite_fabric_5_stream_id = 31;
    // Multi-RISC teardown synchronization stream ID
    static constexpr uint32_t multi_risc_teardown_sync_stream_id = 31;

    static const auto& get_all_stream_ids() {
        static constexpr std::array stream_ids = {
            to_receiver_0_pkts_sent_id,
            to_receiver_1_pkts_sent_id,
            to_receiver_2_pkts_sent_id,
            to_sender_0_pkts_completed_id,
            to_sender_1_pkts_completed_id,
            to_sender_2_pkts_completed_id,
            to_sender_3_pkts_completed_id,
            to_sender_4_pkts_completed_id,
            to_sender_5_pkts_completed_id,
            to_sender_6_pkts_completed_id,
            to_sender_7_pkts_completed_id,
            vc_0_free_slots_from_downstream_edge_1,
            vc_0_free_slots_from_downstream_edge_2,
            vc_0_free_slots_from_downstream_edge_3,
            vc_1_free_slots_from_downstream_edge_1,
            vc_2_free_slots_from_downstream_edge_1,
            vc_2_free_slots_from_downstream_edge_2,
            vc_2_free_slots_from_downstream_edge_3,
            sender_channel_0_free_slots_stream_id,
            sender_channel_1_free_slots_stream_id,
            sender_channel_2_free_slots_stream_id,
            sender_channel_3_free_slots_stream_id,
            sender_channel_4_free_slots_stream_id,
            sender_channel_5_free_slots_stream_id,
            sender_channel_6_free_slots_stream_id,
            sender_channel_7_free_slots_stream_id,
            multi_risc_teardown_sync_stream_id};
        return stream_ids;  // Now 26 stream IDs total (includes VC2)
    }
};

}  // namespace tt::tt_fabric
