// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_edm_packet_header.hpp"
#include "control_channel_interface.hpp"
#include "fabric_context.hpp"
#include <hostdevcommon/fabric_common.h>
#include <chrono>
#include <thread>

namespace tt::tt_fabric {

ControlChannelInterface::ControlChannelInterface(HostToRouterCommInterface* host_to_router_comm_interface) {
    // Initialize host-to-router communication interface
    host_to_router_comm_interface_ptr_ = host_to_router_comm_interface;
}

ControlChannelResult ControlChannelInterface::request_remote_heartbeat_check(
    FabricNodeId& initiator_node,
    chan_id_t initiator_channel,
    FabricNodeId& target_node,
    chan_id_t target_channel,
    uint32_t sequence_id) const {
    if (!host_to_router_comm_interface_ptr_) {
        return ControlChannelResult::OPERATION_FAILED;
    }

    auto packet = create_remote_heartbeat_init_packet(
        initiator_node, initiator_channel, target_node, target_channel, sequence_id);

    return send_control_packet(packet, initiator_node, initiator_channel);
}

bool ControlChannelInterface::check_remote_heartbeat_request_completed(
    FabricNodeId& node_id, chan_id_t eth_chan_id, uint32_t sequence_id) const {
    auto common_fsm_log = host_to_router_comm_interface_ptr_->get_common_fsm_log(node_id, eth_chan_id);
    log_info(tt::LogTest, "Common FSM log sequence id: {}", common_fsm_log.last_processed_sequence_id);
    return common_fsm_log.last_processed_sequence_id == sequence_id;
}

ControlChannelResult ControlChannelInterface::poll_for_remote_heartbeat_request_completion(
    FabricNodeId& node_id,
    chan_id_t eth_chan_id,
    uint32_t sequence_id,
    uint32_t timeout_ms,
    uint32_t poll_interval_ms) const {
    bool request_completed = false;
    const auto sleep_time = std::chrono::milliseconds(poll_interval_ms);
    auto start_time = std::chrono::steady_clock::now();
    while (!request_completed) {
        request_completed = check_remote_heartbeat_request_completed(node_id, eth_chan_id, sequence_id);
        if (!request_completed) {
            log_info(tt::LogTest, "Waiting for request completion...");
            std::this_thread::sleep_for(sleep_time);
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
            if (elapsed > timeout_ms) {
                return ControlChannelResult::OPERATION_FAILED;
            }
        }
    }
    return ControlChannelResult::SUCCESS;
}

ControlChannelResult ControlChannelInterface::send_control_packet(
    ControlPacketHeader& packet, FabricNodeId& target_node, chan_id_t target_channel) const {
    if (!host_to_router_comm_interface_ptr_) {
        return ControlChannelResult::OPERATION_FAILED;
    }

    // Direct packet send - for advanced users
    bool success = host_to_router_comm_interface_ptr_->write_packet_to_router(packet, target_node, target_channel);
    return success ? ControlChannelResult::SUCCESS : ControlChannelResult::BUFFER_FULL;
}

ControlPacketHeader ControlChannelInterface::create_remote_heartbeat_init_packet(
    FabricNodeId& initiator_node,
    chan_id_t initiator_channel,
    FabricNodeId& target_node,
    chan_id_t target_channel,
    uint32_t sequence_id) const {
    ControlPacketHeader packet = {};

    // Basic packet info - triggers RemoteHeartbeatFSM on device
    packet.type = ControlPacketType::REMOTE_HEARTBEAT;
    packet.sub_type = ControlPacketSubType::INIT;

    // Source: Host, Destination: Initiator router (to trigger its FSM)
    // For now, use simple host node ID - this should come from fabric context
    NodeId host_node = {};
    host_node.mesh_id = 0;  // Ignored
    host_node.chip_id = 0;  // Ignored
    packet.src_node_id = host_node;
    packet.dst_node_id.mesh_id = *initiator_node.mesh_id;
    packet.dst_node_id.chip_id = initiator_node.chip_id;
    packet.src_channel_id = eth_chan_magic_values::INVALID_DIRECTION;  // Host channel
    packet.dst_channel_id = initiator_channel;

    packet.context.remote_heartbeat_packet_context.target_node_id.mesh_id = *target_node.mesh_id;
    packet.context.remote_heartbeat_packet_context.target_node_id.chip_id = target_node.chip_id;
    packet.context.remote_heartbeat_packet_context.target_channel_id = target_channel;

    packet.sequence_id = sequence_id;

    return packet;
}

}  // namespace tt::tt_fabric
