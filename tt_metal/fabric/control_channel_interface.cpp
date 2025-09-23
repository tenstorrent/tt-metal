// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "control_channel_interface.hpp"
#include "fabric_context.hpp"
#include <hostdevcommon/fabric_common.h>

namespace tt::tt_fabric {

ControlChannelInterface::ControlChannelInterface(HostToRouterCommInterface* host_to_router_comm_interface) {
    // Initialize host-to-router communication interface
    host_to_router_comm_interface_ptr_ = host_to_router_comm_interface;
}

ControlChannelResult ControlChannelInterface::request_heartbeat_check(
    FabricNodeId& initiator_node, chan_id_t initiator_channel, FabricNodeId& target_node, chan_id_t target_channel) {
    if (!host_to_router_comm_interface_ptr_) {
        return ControlChannelResult::OPERATION_FAILED;
    }

    // Create INIT packet to trigger device HeartbeatFSM
    auto packet = create_heartbeat_init_packet(initiator_node, initiator_channel, target_node, target_channel);

    return send_control_packet(packet, initiator_node, initiator_channel);
}

ControlChannelResult ControlChannelInterface::send_control_packet(
    ControlPacketHeader& packet, FabricNodeId& target_node, chan_id_t target_channel) {
    if (!host_to_router_comm_interface_ptr_) {
        return ControlChannelResult::OPERATION_FAILED;
    }

    // Direct packet send - for advanced users
    bool success = host_to_router_comm_interface_ptr_->write_packet_to_router(packet, target_node, target_channel);
    return success ? ControlChannelResult::SUCCESS : ControlChannelResult::BUFFER_FULL;
}

ControlPacketHeader ControlChannelInterface::create_heartbeat_init_packet(
    FabricNodeId& initiator_node,
    chan_id_t initiator_channel,
    FabricNodeId& target_node,
    chan_id_t target_channel) const {
    ControlPacketHeader packet = {};

    // Basic packet info - triggers HeartbeatFSM on device
    packet.type = ControlPacketType::HEARTBEAT;
    packet.sub_type = ControlPacketSubType::INIT;

    // Source: Host, Destination: Initiator router (to trigger its FSM)
    // For now, use simple host node ID - this should come from fabric context
    NodeId host_node = {};
    host_node.mesh_id = 0;  // Ignored
    host_node.chip_id = 0;  // Ignored
    packet.src_node_id = reinterpret_cast<const NodeId&>(host_node);
    packet.dst_node_id = reinterpret_cast<const NodeId&>(initiator_node);
    packet.src_channel_id = eth_chan_magic_values::INVALID_DIRECTION;  // Host channel
    packet.dst_channel_id = initiator_channel;

    // Context: Tell initiator FSM who to heartbeat (matches device FSM expectations)
    packet.context.heartbeat_packet_context.target_node_id = reinterpret_cast<const NodeId&>(target_node);
    packet.context.heartbeat_packet_context.target_channel_id = target_channel;

    return packet;
}

}  // namespace tt::tt_fabric
