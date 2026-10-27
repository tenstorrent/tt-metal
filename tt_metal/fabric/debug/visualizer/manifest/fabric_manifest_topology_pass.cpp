// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_topology_pass.hpp"

#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt_stl/assert.hpp>
#include <llrt/tt_cluster.hpp>

#include <algorithm>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_describe.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_fields.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"

namespace tt::tt_fabric {

namespace {

using manifest::direction_letter;
using manifest::router_key;

// The router on `peer_chan`, or nullopt when ControlPlane has no active router there (the channel was trimmed from
// the peer's routing planes).
std::optional<manifest::RouterRef> find_peer_router(
    const ControlPlane& control_plane, FabricNodeId peer_node, chan_id_t peer_chan) {
    for (const auto& [chan, direction] : control_plane.get_active_fabric_eth_channels(peer_node)) {
        if (chan == peer_chan) {
            return manifest::RouterRef{
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
chan_id_t get_sibling_chan(
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

tt::tt_metal::CoreCoord get_router_logical_core(const tt::Cluster& cluster, ChipId physical_chip_id, chan_id_t chan) {
    const auto core = cluster.get_soc_desc(physical_chip_id).get_eth_core_for_channel(chan, CoordSystem::LOGICAL);
    return tt::tt_metal::CoreCoord(core.x, core.y);
}

tt::tt_metal::CoreCoord get_router_virtual_core(const tt::Cluster& cluster, ChipId physical_chip_id, chan_id_t chan) {
    return cluster.get_virtual_coordinate_from_logical_coordinates(
        physical_chip_id, get_router_logical_core(cluster, physical_chip_id, chan), CoreType::ETH);
}

// The value of the flag field keyed `key`.
bool read_flag_field(const std::vector<manifest::Field>& fields, std::string_view key) {
    const auto it = std::ranges::find(fields, key, &manifest::Field::key);
    TT_FATAL(it != fields.end(), "Fabric manifest: no field keyed {}", key);
    const auto* flag = it->content.has_value() ? std::get_if<manifest::content::Flag>(&*it->content) : nullptr;
    TT_FATAL(flag != nullptr, "Fabric manifest: field {} is not a flag", key);
    return flag->value;
}

// Resolves each collected router's identity from its Ethernet channel: its key (direction and routing plane, ie. E0)
// and cores. Returns the routers in the order ControlPlane lists the chip's active channels.
std::vector<manifest::Router> resolve_router_identities(
    std::vector<manifest::Router> collected,
    const ControlPlane& control_plane,
    const tt::Cluster& cluster,
    FabricNodeId node,
    ChipId physical_chip_id) {
    // Index the collected routers by their Ethernet channel
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
        // Get the router for this channel
        const auto it = by_chan.find(chan);
        TT_FATAL(
            it != by_chan.end(),
            "Fabric manifest: {} channel {} is an active fabric router, but no router was collected for it",
            node,
            chan);
        manifest::Router router = std::move(it->second);
        by_chan.erase(it);

        // Fill in the router's identity
        auto& identity = router.identity;
        identity.direction = direction;
        identity.routing_plane = control_plane.get_routing_plane_id(node, chan);
        TT_FATAL(
            keys.emplace(identity.direction, identity.routing_plane).second,
            "Fabric manifest: {} has two routers keyed {}",
            node,
            router_key(identity.direction, identity.routing_plane));
        identity.logical_core = get_router_logical_core(cluster, physical_chip_id, chan);
        identity.virtual_core = get_router_virtual_core(cluster, physical_chip_id, chan);
        routers.push_back(std::move(router));
    }

    // Ensure none of the collected routers haven't been resolved
    TT_FATAL(
        by_chan.empty(),
        "Fabric manifest: {} has {} collected routers on channels that are not active fabric routers",
        node,
        by_chan.size());

    return routers;
}

// The router that leads the chip's local sync.
manifest::RouterRef get_master_router(
    const std::vector<manifest::Router>& routers, FabricNodeId node, uint32_t master_eth_chan) {
    const auto it = std::ranges::find(
        routers, master_eth_chan, [](const manifest::Router& router) { return router.identity.eth_chan; });
    TT_FATAL(
        it != routers.end(),
        "Fabric manifest: {}'s local sync master is channel {}, which has no router",
        node,
        master_eth_chan);
    return manifest::RouterRef{
        .node = node, .direction = it->identity.direction, .routing_plane = it->identity.routing_plane};
}

// Resolves the router's link to the peer router on its other end, and whether the link crosses hosts or wraps.
void resolve_link_peer(
    manifest::Router& router,
    const ControlPlane& control_plane,
    FabricType fabric_type,
    FabricNodeId node,
    ChipId physical_chip_id) {
    const chan_id_t chan = router.identity.eth_chan;
    const auto peer = control_plane.try_get_connected_mesh_chip_chan_ids(node, chan);
    if (peer.has_value()) {
        router.link.peer = find_peer_router(control_plane, peer->first, peer->second);
    }
    router.link.cross_host = control_plane.is_cross_host_eth_link(physical_chip_id, chan);
    router.link.wrap = is_wrap_link(fabric_type, control_plane.get_mesh_graph(), node, router.identity.direction, peer);
}

// Resolves each downstream edge to the sibling router it lands on.
void resolve_edge_siblings(
    std::vector<manifest::Router>& routers, const ControlPlane& control_plane, FabricNodeId node) {
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
                const auto target = get_sibling_chan(control_plane, node, chan, edge.target.direction);
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
                        !read_flag_field(sender.fields, manifest::k_static_connection_key),
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

// Resolves the Ethernet channels the builders recorded to the routers they name: each router's key and cores, the
// local sync master, each link's peer, and each edge's sibling.
manifest::Chip resolve_chip_routers(
    manifest::Chip chip,
    const ControlPlane& control_plane,
    const tt::Cluster& cluster,
    FabricType fabric_type,
    FabricNodeId node,
    ChipId physical_chip_id) {
    chip.routers = resolve_router_identities(std::move(chip.routers), control_plane, cluster, node, physical_chip_id);
    if (chip.local_sync.has_value()) {
        chip.local_sync->master = get_master_router(chip.routers, node, chip.local_sync->master_eth_chan);
    }
    for (auto& router : chip.routers) {
        resolve_link_peer(router, control_plane, fabric_type, node, physical_chip_id);
    }
    resolve_edge_siblings(chip.routers, control_plane, node);
    return chip;
}

manifest::Chip describe_chip(
    const ControlPlane& control_plane,
    const tt::Cluster& cluster,
    const FabricBuilderContext& builder_context,
    FabricType fabric_type,
    FabricNodeId node) {
    // Get the chip's coordinates in its mesh
    const MeshCoordinate mesh_coord = control_plane.get_mesh_graph().chip_to_coordinate(node.mesh_id, node.chip_id);
    std::vector<uint32_t> coords(mesh_coord.coords().begin(), mesh_coord.coords().end());

    // Non-local chips, where locality is at the host rank granularity, can only be described by their place in the
    // mesh. The chip's owning rank takes care of more in depth description.
    const auto physical_chip_id = control_plane.try_get_physical_chip_id_from_fabric_node_id(node);
    if (!physical_chip_id.has_value()) {
        return manifest::Chip{.fabric_node_id = node, .mesh_coord = std::move(coords)};
    }

    // A chip that built no routers published nothing, so it starts with no routers
    manifest::Chip chip = builder_context.has_manifest_chip(*physical_chip_id)
                              ? builder_context.get_manifest_chip(*physical_chip_id)
                              : manifest::Chip{};

    // Fill in the chip's details
    chip.fabric_node_id = node;
    chip.mesh_coord = std::move(coords);
    chip.physical_chip_id = physical_chip_id;
    chip.asic_id = *control_plane.get_asic_id_from_fabric_node_id(node);

    // Resolve the chip's routers against the control plane and the chip's other routers
    return resolve_chip_routers(std::move(chip), control_plane, cluster, fabric_type, node, *physical_chip_id);
}

manifest::Mesh describe_mesh(
    const ControlPlane& control_plane,
    const tt::Cluster& cluster,
    const FabricBuilderContext& builder_context,
    FabricType fabric_type,
    MeshId mesh_id) {
    // Get the mesh's shape and configuration
    const auto& mesh_graph = control_plane.get_mesh_graph();
    const MeshShape mesh_shape = mesh_graph.get_mesh_shape(mesh_id);
    manifest::Mesh mesh{.mesh_id = mesh_id, .express_routing = control_plane.express_routing_enabled(mesh_id)};
    for (size_t dim = 0; dim < mesh_shape.dims(); ++dim) {
        mesh.shape.push_back(mesh_shape[dim]);
    }

    // has_genuine_torus_axis() is defined only for a 2D shape, and deliberately reports false for a declared
    // torus axis whose extent is too small to realize a distinct wrap edge.
    if (mesh_shape.dims() == 2) {
        mesh.torus = manifest::MeshTorus{
            .y = has_genuine_torus_axis(fabric_type, mesh_shape, 0),
            .x = has_genuine_torus_axis(fabric_type, mesh_shape, 1),
        };
    }

    // The builder context plans credits only for the meshes on this host
    const auto local_mesh_ids = control_plane.get_local_mesh_id_bindings();
    if (std::ranges::find(local_mesh_ids, mesh_id) != local_mesh_ids.end()) {
        mesh.credit_transport = manifest::describe_credit_transport(builder_context, mesh_id);
    }

    // Describe each chip in the mesh
    for (const auto& [_, fabric_chip_id] : mesh_graph.get_chip_ids(mesh_id)) {
        mesh.chips.push_back(
            describe_chip(control_plane, cluster, builder_context, fabric_type, FabricNodeId(mesh_id, fabric_chip_id)));
    }

    return mesh;
}

}  // namespace

namespace manifest {

std::vector<Mesh> describe_meshes(
    const ControlPlane& control_plane, const tt::Cluster& cluster, const FabricBuilderContext& builder_context) {
    const FabricType fabric_type = get_fabric_type(control_plane.get_fabric_config(), cluster.is_ubb_galaxy());
    auto mesh_ids = control_plane.get_mesh_graph().get_all_mesh_ids();
    std::ranges::sort(mesh_ids, {}, [](const MeshId& mesh_id) { return *mesh_id; });
    std::vector<Mesh> meshes;
    meshes.reserve(mesh_ids.size());
    for (const auto& mesh_id : mesh_ids) {
        meshes.push_back(describe_mesh(control_plane, cluster, builder_context, fabric_type, mesh_id));
    }
    return meshes;
}

}  // namespace manifest

}  // namespace tt::tt_fabric
