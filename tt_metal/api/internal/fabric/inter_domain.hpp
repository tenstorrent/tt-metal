// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <vector>

#include <hostdevcommon/fabric_common.h>
#include <tt-metalium/vocabulary/core_coord.hpp>
#include <tt-metalium/workload/program.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>

namespace tt::tt_metal::distributed {
class MeshDevice;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_fabric::internal {

struct InterDomainPort {
    FabricNodeId node{MeshId{0}, 0};
    chan_id_t eth_chan = 0;
    RoutingDirection direction = RoutingDirection::N;
    uint32_t plane = 0;
    bool link_trained = true;
    bool has_router = true;
    uint64_t peer_asic = 0;
    chan_id_t peer_chan = 0;
    int context = 0;
};

// What two peers swap before they connect.
struct InterDomainMetadata {
    uint32_t domain_id = 0;
    uint32_t vc2_receiver_channel_addr = 0;
    uint32_t vc2_receiver_channel_slots = 0;
    uint32_t max_packet_size_bytes = 0;
    uint32_t layout_fingerprint = 0;
    uint32_t completion_region_base = 0;
    uint32_t assigned_domain_credit_slot = 0;
    uint32_t flows_per_domain = 0;
};

// This host's inter-domain ports on the fabric `mesh_device` runs on.
std::vector<InterDomainPort> get_inter_domain_ports(const tt::tt_metal::distributed::MeshDevice& mesh_device);

uint64_t get_inter_domain_port_asic(const InterDomainPort& port);

InterDomainMetadata get_inter_domain_metadata(const InterDomainPort& port, uint32_t domain_id);

InterDomainState get_inter_domain_state(const InterDomainPort& port);

// Both sides must connect. Throws on a failed connect, leaving the router in DRAIN.
void connect_inter_domain(
    const InterDomainPort& port,
    uint32_t domain_id,
    const InterDomainMetadata& peer,
    std::chrono::milliseconds timeout = std::chrono::seconds(30));

// Waits for the peer's disconnect unless `force` or past the timeout. Returns with the router in DRAIN.
void disconnect_inter_domain(
    const InterDomainPort& port, bool force = false, std::chrono::milliseconds timeout = std::chrono::seconds(30));

// Read on device with WorkerToFabricEdmSender::build_from_args.
void append_inter_domain_connection_rt_args(
    const InterDomainPort& port,
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreCoord& worker_core,
    std::vector<uint32_t>& rt_args);

}  // namespace tt::tt_fabric::internal
