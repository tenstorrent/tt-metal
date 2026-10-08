// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_chip_pass.hpp"

#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt_stl/assert.hpp>
#include <llrt/tt_cluster.hpp>

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_fields.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"

namespace tt::tt_fabric {

namespace {

using manifest::direction_letter;
using manifest::router_key;

// The router on `peer_chan`, or nullopt when ControlPlane has no active router there (the channel was trimmed from
// the peer's routing planes).
std::optional<manifest::PeerRouterRef> peer_router_ref(
    const ControlPlane& control_plane, FabricNodeId peer_node, chan_id_t peer_chan) {
    for (const auto& [chan, direction] : control_plane.get_active_fabric_eth_channels(peer_node)) {
        if (chan == peer_chan) {
            return manifest::PeerRouterRef{
                .node = peer_node,
                .direction = direction,
                .routing_plane = control_plane.get_routing_plane_id(peer_node, chan),
            };
        }
    }
    return std::nullopt;
}

// Wrap edges are resolved here rather than inferred by the viewer: a link wraps when its axis genuinely
// closes and its coordinate delta spans the mesh.
bool is_wrap_link(
    FabricType fabric_type,
    const MeshGraph& mesh_graph,
    FabricNodeId node,
    eth_chan_directions direction,
    const std::optional<std::pair<FabricNodeId, chan_id_t>>& peer) {
    const MeshShape mesh_shape = mesh_graph.get_mesh_shape(node.mesh_id);
    const bool is_east_west = direction == eth_chan_directions::EAST || direction == eth_chan_directions::WEST;
    const bool is_north_south = direction == eth_chan_directions::NORTH || direction == eth_chan_directions::SOUTH;

    // Wrap links require a 2D mesh, a peer in the same mesh, and an east-west or north-south direction
    if (mesh_shape.dims() != 2 || !peer.has_value() || peer->first.mesh_id != node.mesh_id ||
        !(is_east_west || is_north_south)) {
        return false;
    }
    const uint32_t axis = is_east_west ? 1 : 0;
    if (!has_genuine_torus_axis(fabric_type, mesh_shape, axis)) {
        return false;
    }
    const uint32_t here = mesh_graph.chip_to_coordinate(node.mesh_id, node.chip_id)[axis];
    const uint32_t there = mesh_graph.chip_to_coordinate(node.mesh_id, peer->first.chip_id)[axis];
    const uint32_t delta = here > there ? here - there : there - here;
    return delta == mesh_shape[axis] - 1;
}

// The channel of the router facing `direction` on the same chip and routing plane as the router on `chan`.
chan_id_t sibling_chan(
    const ControlPlane& control_plane, FabricNodeId node, chan_id_t chan, eth_chan_directions direction) {
    const auto plane = control_plane.get_routing_plane_id(node, chan);
    for (const auto& [sibling, sibling_direction] : control_plane.get_active_fabric_eth_channels(node)) {
        if (sibling_direction == direction && control_plane.get_routing_plane_id(node, sibling) == plane) {
            return sibling;
        }
    }
    TT_THROW(
        "Fabric manifest: {} channel {} has a sibling facing {} on routing plane {}, but no router is there",
        node,
        chan,
        direction_letter(direction),
        plane);
}

tt::tt_metal::CoreCoord router_logical_core(const tt::Cluster& cluster, ChipId physical_chip_id, chan_id_t chan) {
    const auto core = cluster.get_soc_desc(physical_chip_id).get_eth_core_for_channel(chan, CoordSystem::LOGICAL);
    return tt::tt_metal::CoreCoord(core.x, core.y);
}

tt::tt_metal::CoreCoord router_virtual_core(const tt::Cluster& cluster, ChipId physical_chip_id, chan_id_t chan) {
    return cluster.get_virtual_coordinate_from_logical_coordinates(
        physical_chip_id, router_logical_core(cluster, physical_chip_id, chan), CoreType::ETH);
}

// The value of the flag field keyed `key`.
bool flag_field(const std::vector<manifest::Field>& fields, std::string_view key) {
    const auto it = std::ranges::find(fields, key, &manifest::Field::key);
    TT_FATAL(it != fields.end(), "Fabric manifest: no field keyed {}", key);
    const auto* flag = std::get_if<manifest::content::Flag>(&it->content);
    TT_FATAL(flag != nullptr, "Fabric manifest: field {} is not a flag", key);
    return flag->value;
}

// The collected routers in the order ControlPlane lists the chip's active channels, each with its key and cores.
std::vector<manifest::Router> key_routers(
    std::vector<manifest::Router> collected,
    const ControlPlane& control_plane,
    const tt::Cluster& cluster,
    FabricNodeId node,
    ChipId physical_chip_id) {
    std::map<chan_id_t, manifest::Router> by_chan;
    for (auto& router : collected) {
        const chan_id_t chan = router.identity.eth_chan;
        TT_FATAL(
            by_chan.emplace(chan, std::move(router)).second,
            "Fabric manifest: {} has two collected routers on channel {}",
            node,
            chan);
    }

    std::vector<manifest::Router> routers;
    std::set<std::pair<eth_chan_directions, routing_plane_id_t>> keys;
    for (const auto& [chan, direction] : control_plane.get_active_fabric_eth_channels(node)) {
        const auto it = by_chan.find(chan);
        TT_FATAL(
            it != by_chan.end(),
            "Fabric manifest: {} channel {} is an active fabric router, but no router was collected for it",
            node,
            chan);
        manifest::Router router = std::move(it->second);
        by_chan.erase(it);

        auto& identity = router.identity;
        identity.direction = direction;
        identity.routing_plane = control_plane.get_routing_plane_id(node, chan);
        TT_FATAL(
            keys.emplace(identity.direction, identity.routing_plane).second,
            "Fabric manifest: {} has two routers keyed {}",
            node,
            router_key(identity.direction, identity.routing_plane));
        identity.logical_core = router_logical_core(cluster, physical_chip_id, chan);
        identity.virtual_core = router_virtual_core(cluster, physical_chip_id, chan);
        routers.push_back(std::move(router));
    }
    TT_FATAL(
        by_chan.empty(),
        "Fabric manifest: {} has {} collected routers on channels that are not active fabric routers",
        node,
        by_chan.size());
    return routers;
}

void join_link(
    manifest::Router& router,
    const ControlPlane& control_plane,
    FabricType fabric_type,
    FabricNodeId node,
    ChipId physical_chip_id) {
    const chan_id_t chan = router.identity.eth_chan;
    const auto peer = control_plane.try_get_connected_mesh_chip_chan_ids(node, chan);
    if (peer.has_value()) {
        router.link.peer = peer_router_ref(control_plane, peer->first, peer->second);
    }
    router.link.cross_host = control_plane.is_cross_host_eth_link(physical_chip_id, chan);
    router.link.wrap = is_wrap_link(fabric_type, control_plane.get_mesh_graph(), node, router.identity.direction, peer);
}

// Resolves each edge to its sibling.
void join_edges(std::vector<manifest::Router>& routers, const ControlPlane& control_plane, FabricNodeId node) {
    std::map<chan_id_t, const manifest::Router*> by_chan;
    for (const auto& router : routers) {
        by_chan.emplace(router.identity.eth_chan, &router);
    }

    struct Landing {
        const manifest::Router* router;
        bool through_tensix_mux;
    };
    // (sibling's channel, VC, sender channel) to the edge that lands there.
    std::map<std::tuple<chan_id_t, uint32_t, uint32_t>, Landing> landed_by;
    for (auto& router : routers) {
        const chan_id_t chan = router.identity.eth_chan;
        for (uint32_t vc = 0; vc < router.intra_chip_downstream_edges.size(); ++vc) {
            for (auto& edge : router.intra_chip_downstream_edges[vc]) {
                const auto target = sibling_chan(control_plane, node, chan, edge.target.direction);
                const auto it = by_chan.find(target);
                TT_FATAL(
                    it != by_chan.end(),
                    "Fabric manifest: {} channel {} has an edge to channel {}, which was not collected",
                    node,
                    chan,
                    target);
                const auto& sibling = *it->second;
                const auto num_senders = sibling.shape.senders_per_vc.at(vc);
                TT_FATAL(
                    edge.landing_channel < num_senders,
                    "Fabric manifest: {} channel {}'s VC{} edge {} lands on channel {}, but channel {} has {}",
                    node,
                    chan,
                    vc,
                    edge.edge,
                    edge.landing_channel,
                    target,
                    num_senders);
                edge.through_tensix_mux = edge.core != sibling.identity.virtual_core;
                const auto [landed, inserted] = landed_by.emplace(
                    std::tuple{target, vc, edge.landing_channel},
                    Landing{.router = &router, .through_tensix_mux = edge.through_tensix_mux});
                TT_FATAL(
                    inserted,
                    "Fabric manifest: {} channels {} and {} both have an edge into channel {}'s VC{} sender {}",
                    node,
                    landed->second.router->identity.eth_chan,
                    chan,
                    target,
                    vc,
                    edge.landing_channel);
            }
        }
    }

    for (auto& router : routers) {
        const chan_id_t chan = router.identity.eth_chan;
        for (uint32_t vc = 0; vc < router.channels.senders.size(); ++vc) {
            for (uint32_t ch = 0; ch < router.channels.senders[vc].size(); ++ch) {
                auto& sender = router.channels.senders[vc][ch];
                const auto landed = landed_by.find({chan, vc, ch});
                if (landed == landed_by.end()) {
                    TT_FATAL(
                        !flag_field(sender.fields, manifest::k_static_connection_key),
                        "Fabric manifest: {} channel {}'s VC{} sender {} waits for a sibling to connect, but no "
                        "sibling's edge lands on it",
                        node,
                        chan,
                        vc,
                        ch);
                    continue;
                }
                if (landed->second.through_tensix_mux) {
                    continue;
                }
                TT_FATAL(
                    !sender.producer.has_value(),
                    "Fabric manifest: {} channel {}'s VC{} sender {} is fed by the local worker or tensix mux, but "
                    "channel {}'s edge lands on it",
                    node,
                    chan,
                    vc,
                    ch,
                    landed->second.router->identity.eth_chan);
                sender.producer = manifest::SiblingRouterRef{.direction = landed->second.router->identity.direction};
            }
        }
    }
}

}  // namespace

manifest::Chip join_chip(
    manifest::Chip chip,
    const ControlPlane& control_plane,
    const tt::Cluster& cluster,
    FabricType fabric_type,
    FabricNodeId node,
    ChipId physical_chip_id) {
    chip.routers = key_routers(std::move(chip.routers), control_plane, cluster, node, physical_chip_id);
    for (auto& router : chip.routers) {
        join_link(router, control_plane, fabric_type, node, physical_chip_id);
    }
    join_edges(chip.routers, control_plane, node);
    return chip;
}

}  // namespace tt::tt_fabric
