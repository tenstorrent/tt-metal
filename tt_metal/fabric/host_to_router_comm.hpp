// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fabric/fabric_edm_packet_header.hpp"
#include "host_to_router_comm_helpers.hpp"
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

private:
    uint32_t num_buffer_slots_;
    uint32_t remote_buffer_address_;
    uint32_t remote_read_counter_address_;
    uint32_t remote_write_counter_address_;

    uint32_t get_remote_read_counter(FabricNodeId& node_id, chan_id_t eth_chan_id) const;
    bool has_space_for_packet(HostChannelCounter& local_write_counter, uint32_t remote_read_counter) const;
    uint32_t get_next_buffer_address(HostChannelCounter& local_write_counter) const;
    void update_remote_write_counter(FabricNodeId& node_id, chan_id_t eth_chan_id, uint32_t local_write_counter) const;
};

}  // namespace tt::tt_fabric
