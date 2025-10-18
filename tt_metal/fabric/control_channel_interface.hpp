// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fabric_edm_packet_header.hpp"
#include "host_to_router_comm.hpp"
#include <hostdevcommon/fabric_common.h>
#include <memory>
#include <vector>

namespace tt::tt_fabric {

// Forward declarations
class FabricNodeId;
class FabricContext;

// Result types for error handling
enum class ControlChannelResult : uint8_t {
    SUCCESS = 0,
    BUFFER_FULL = 1,
    NODE_UNREACHABLE = 2,
    INVALID_CHANNEL = 3,
    OPERATION_FAILED = 4
};

// Host-side control channel interface - communicates with device-side FabricControlChannel
class ControlChannelInterface {
public:
    ControlChannelInterface() = default;
    ~ControlChannelInterface() = default;

    ControlChannelInterface(HostToRouterCommInterface* host_to_router_comm_interface);

    // Host-initiated control protocols
    ControlChannelResult request_remote_heartbeat_check(
        FabricNodeId& initiator_node,
        chan_id_t initiator_channel,
        FabricNodeId& target_node,
        chan_id_t target_channel,
        uint32_t sequence_id) const;

    bool check_remote_heartbeat_request_completed(
        FabricNodeId& node_id, chan_id_t eth_chan_id, uint32_t sequence_id) const;

    ControlChannelResult poll_for_remote_heartbeat_request_completion(
        FabricNodeId& node_id,
        chan_id_t eth_chan_id,
        uint32_t sequence_id,
        uint32_t timeout_ms = 5000,
        uint32_t poll_interval_ms = 100) const;

    // Low-level packet sending (for advanced users)
    ControlChannelResult send_control_packet(
        ControlPacketHeader& packet, FabricNodeId& target_node, chan_id_t target_channel) const;

private:
    // Packet construction helpers
    ControlPacketHeader create_remote_heartbeat_init_packet(
        FabricNodeId& initiator_node,
        chan_id_t initiator_channel,
        FabricNodeId& target_node,
        chan_id_t target_channel,
        uint32_t sequence_id) const;

    HostToRouterCommInterface* host_to_router_comm_interface_ptr_;
};

}  // namespace tt::tt_fabric
