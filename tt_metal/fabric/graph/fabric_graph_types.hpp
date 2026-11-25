// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <compare>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <hostdevcommon/fabric_common.h>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt_stl/strong_type.hpp>

#include "tt_metal/fabric/builder/fabric_edge_capability.hpp"

namespace tt::tt_fabric::graph {

// N/E/S/W/Z only, never C or NONE
using PortDirection = ttsl::StrongType<RoutingDirection, struct PortDirectionTag>;

// One port of a chip: a direction and routing plane, linked to `peer`.
struct Port {
    FabricNodeId node{MeshId{0}, 0};
    PortDirection direction{RoutingDirection::N};
    uint32_t plane = 0;
    FabricNodeId peer{MeshId{0}, 0};
    EdgeCapability capability = EdgeCapability::INTRAMESH_CARDINAL;
    // perimeter port whose link is down: its router relays all it sends to this side's router on relay_plane
    std::optional<uint32_t> relay_plane;
};

struct Vc {
    uint32_t id = 0;
    std::string name;
};

struct PerimeterPort {
    FabricNodeId node{MeshId{0}, 0};
    chan_id_t eth_chan = 0;
    PortDirection direction{RoutingDirection::N};
    uint32_t plane = 0;
    bool link_trained = true;
    uint64_t peer_asic = 0;
    chan_id_t peer_chan = 0;
};

using RouterIndex = uint32_t;
using ChannelIndex = uint32_t;

// One router per port.
struct LogicalRouter {
    std::string id;
    Port port;
    std::optional<RouterIndex> peer;  // router at the other end of the link
    std::vector<ChannelIndex> channels;
};

enum class ChannelKind : uint8_t { Sender, Receiver };

struct LogicalChannel {
    RouterIndex router = 0;
    uint32_t vc = 0;
    ChannelKind kind = ChannelKind::Sender;
    bool injection = false;
    std::optional<PortDirection> from;  // forwarding sender: where its traffic came in
};

enum class Transport : uint8_t {
    Eth,
    Noc,
    Dynamic,
};

struct LogicalEdge {
    ChannelIndex from = 0;
    ChannelIndex to = 0;
    Transport transport = Transport::Eth;
};

struct Ring {
    std::string id;
    uint32_t vc = 0;
    uint32_t plane = 0;
    std::vector<RouterIndex> members;
};

struct LogicalGraph {
    std::vector<Vc> vcs;
    std::vector<LogicalRouter> routers;
    std::vector<LogicalChannel> channels;
    std::vector<LogicalEdge> edges;
    std::vector<Ring> rings;

    std::optional<RouterIndex> find_router(std::string_view id) const;
    std::optional<ChannelIndex> receiver(RouterIndex router, uint32_t vc) const;
    std::vector<ChannelIndex> channels_of(RouterIndex router, ChannelKind kind) const;
};

// "m<mesh>.c<chip>.<direction>.p<plane>"
std::string router_id(const FabricNodeId& node, PortDirection direction, uint32_t plane);

// "rx", "worker", or the direction its traffic came in on (a relay sender: its own)
std::string channel_name(const LogicalChannel& channel);

// A perimeter router's relay sender: it takes what a same-side router on another plane relays, so its traffic comes
// in from the router's own side.
bool is_relay_sender(const LogicalGraph& graph, const LogicalChannel& channel);

}  // namespace tt::tt_fabric::graph
