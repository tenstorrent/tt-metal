// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// EXPERIMENTAL: inter-domain fabric (TT_METAL_FABRIC_INTER_DOMAIN=1, graph builder, 2D).
//
// A domain's fabric puts an inter-domain router on every live link that leaves the domain. It boots
// into DRAIN. The host binds its peer at run time: each side reads its router's metadata, the two
// sides swap it out of band, then each connects its own router with the peer's. Workers then send
// over the router's VC2 worker channel into the peer domain. Disconnect puts it back in DRAIN.

#include <chrono>
#include <cstdint>
#include <vector>

#include <hostdevcommon/fabric_common.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>

namespace tt::tt_fabric::experimental {

// A link that leaves this domain, with its inter-domain router.
struct InterDomainPort {
    FabricNodeId node{MeshId{0}, 0};
    chan_id_t eth_chan = 0;
    RoutingDirection direction = RoutingDirection::N;
    uint32_t plane = 0;
    bool link_trained = true;  // false: down at build; connect on it fails
    bool has_router = true;    // false on a dead link, and (not yet) on Z
    uint64_t peer_asic = 0;    // the chip and channel across the link
    chan_id_t peer_chan = 0;
};

// What two peers swap before they connect (the design's SentCrossDomainConnectionMetadata).
struct InterDomainMetadata {
    uint32_t domain_id = 0;
    uint32_t vc2_receiver_channel_addr = 0;
    uint32_t vc2_receiver_channel_slots = 0;
    uint32_t max_packet_size_bytes = 0;
    uint32_t layout_fingerprint = 0;  // the router L1 layout connect/disconnect write across the link; must match
    // completions are not wired up yet
    uint32_t completion_region_base = 0;
    uint32_t assigned_domain_credit_slot = 0;
    uint32_t flows_per_domain = 0;
};

// This host's inter-domain ports, from the running fabric.
std::vector<InterDomainPort> get_inter_domain_ports();

// The chip a port is on, by its unique (asic) id.
uint64_t get_inter_domain_port_asic(const InterDomainPort& port);

// This side's metadata for `port`, for the peer.
InterDomainMetadata get_inter_domain_metadata(const InterDomainPort& port, uint32_t domain_id);

// The router's lifecycle state.
InterDomainState get_inter_domain_state(const InterDomainPort& port);

// Connects the port's router to the peer described by `peer`. Both sides must connect; returns once
// connected. Throws on a failed connect (bad metadata, an untrained link, or no peer before the
// timeout, which aborts the attempt and leaves the router in DRAIN).
void connect_inter_domain(
    const InterDomainPort& port,
    uint32_t domain_id,
    const InterDomainMetadata& peer,
    std::chrono::milliseconds timeout = std::chrono::seconds(30));

// Disconnects the port's router. Healthy: waits for the peer's disconnect; past the timeout, or with
// `force`, resets without it. Returns once the router is back in DRAIN.
void disconnect_inter_domain(
    const InterDomainPort& port, bool force = false, std::chrono::milliseconds timeout = std::chrono::seconds(30));

// Runtime args for a worker on `worker_core` (on the port's chip) to send into the peer domain through
// the port's router, whose senders feed the peer's VC2 receiver. Read on device with
// WorkerToFabricEdmSender::build_from_args, like any fabric connection.
void append_inter_domain_connection_rt_args(
    const InterDomainPort& port,
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreCoord& worker_core,
    std::vector<uint32_t>& rt_args);

}  // namespace tt::tt_fabric::experimental
