// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fabric/fabric_edm_packet_header.hpp"
#include "host_to_router_comm_helpers.hpp"
#include "fabric_context.hpp"
#include <hostdevcommon/fabric_common.h>
#include <cstdint>
#include <cstddef>

namespace tt::tt_fabric {

// forward declaration
class FabricNodeId;

class HostToRouterCommInterface {
public:
    HostToRouterCommInterface();
    ~HostToRouterCommInterface() = default;

    bool write_packet_to_router(ControlPacketHeader& packet, FabricNodeId& node_id, chan_id_t eth_chan_id);

    CommonFSMLog get_common_fsm_log(FabricNodeId& node_id, chan_id_t eth_chan_id) const;

private:
    HostToRouterCommConfig* host_to_router_comm_config_ptr_;

    uint32_t get_router_read_counter(FabricNodeId& node_id, chan_id_t eth_chan_id) const;
    bool has_space_for_packet(
        FabricNodeId& node_id, chan_id_t eth_chan_id, HostChannelCounter& local_write_counter) const;
    uint32_t get_next_buffer_address(HostChannelCounter& local_write_counter) const;
    void update_router_write_counter(FabricNodeId& node_id, chan_id_t eth_chan_id, uint32_t local_write_counter) const;
};

}  // namespace tt::tt_fabric
