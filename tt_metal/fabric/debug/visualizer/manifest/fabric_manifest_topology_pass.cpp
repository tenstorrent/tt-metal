// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_topology_pass.hpp"

#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt_stl/assert.hpp>
#include <llrt/tt_cluster.hpp>

#include <algorithm>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_describe.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_published_chips.hpp"

namespace tt::tt_fabric {

namespace {

using manifest::router_key;

// The router on `peer_chan`, or nullopt when the peer's builder built none there. The peer may be on another host, so
// this asks ControlPlane for the channels FabricBuilder builds routers on (its routing planes in the direction).
std::optional<manifest::RouterRef> find_peer_router(
    const ControlPlane& control_plane, FabricNodeId peer_node, chan_id_t peer_chan) {
    for (const auto& [chan, direction] : control_plane.get_active_fabric_eth_channels(peer_node)) {
        // Find the peer ethernet channel in the active channels list
        if (chan != peer_chan) {
            continue;
        }

        // Find the peer's routing plane in the direction it faces. If no active
        // routing planes are active in that direction, the peer has no router there.
        const auto routed_chans = control_plane.get_active_fabric_eth_routing_planes_in_direction(
            peer_node, control_plane.eth_direction_to_routing_direction(direction));
        if (std::ranges::find(routed_chans, chan) == routed_chans.end()) {
            return std::nullopt;
        }

        return manifest::RouterRef{
            .node = peer_node,
            .direction = direction,
            .routing_plane = control_plane.get_routing_plane_id(peer_node, chan),
        };
    }
    return std::nullopt;
}

// Which axes of `mesh_id` MeshGraph wrapped, or null for a mesh that isn't 2D. MeshGraph builds each mesh's connections
// from its own fabric type (get_valid_connections), and gives the corner chip (0, 0) an N edge only when the mesh wraps
// rows and a W edge only when it wraps columns.
std::optional<manifest::MeshTorus> describe_mesh_torus(const MeshGraph& mesh_graph, MeshId mesh_id) {
    // No torus for non-2D meshes
    if (mesh_graph.get_mesh_shape(mesh_id).dims() != 2) {
        return std::nullopt;
    }

    const auto& corner_edges = mesh_graph.get_intra_mesh_connectivity().at(*mesh_id).at(0);
    // Check if any of the corner chip's edges have connections to the wrapping direction
    const auto has_edge_toward = [&](RoutingDirection direction) {
        return std::ranges::any_of(
            corner_edges, [&](const auto& edge) { return edge.second.port_direction == direction; });
    };
    return manifest::MeshTorus{.y = has_edge_toward(RoutingDirection::N), .x = has_edge_toward(RoutingDirection::W)};
}

// Wrap edges are resolved here rather than inferred by the viewer: a link wraps when its mesh wraps the link's axis
// and its coordinate delta spans the mesh.
bool is_wrap_link(
    const std::optional<manifest::MeshTorus>& torus,
    const MeshGraph& mesh_graph,
    FabricNodeId node,
    eth_chan_directions direction,
    const std::optional<std::pair<FabricNodeId, chan_id_t>>& peer) {
    const MeshShape mesh_shape = mesh_graph.get_mesh_shape(node.mesh_id);
    const bool is_east_west = direction == eth_chan_directions::EAST || direction == eth_chan_directions::WEST;
    const bool is_north_south = direction == eth_chan_directions::NORTH || direction == eth_chan_directions::SOUTH;

    // Wrap links require a 2D mesh, a peer in the same mesh, and an east-west or north-south direction
    if (!torus.has_value() || !peer.has_value() || peer->first.mesh_id != node.mesh_id ||
        !(is_east_west || is_north_south)) {
        return false;
    }
    const uint32_t axis = is_east_west ? 1 : 0;
    if (!(is_east_west ? torus->x : torus->y)) {
        return false;
    }
    const uint32_t here = mesh_graph.chip_to_coordinate(node.mesh_id, node.chip_id)[axis];
    const uint32_t there = mesh_graph.chip_to_coordinate(node.mesh_id, peer->first.chip_id)[axis];
    const uint32_t delta = here > there ? here - there : there - here;
    return delta == mesh_shape[axis] - 1;
}

tt::tt_metal::CoreCoord get_router_logical_core(const tt::Cluster& cluster, ChipId physical_chip_id, chan_id_t chan) {
    const auto core = cluster.get_soc_desc(physical_chip_id).get_eth_core_for_channel(chan, CoordSystem::LOGICAL);
    return tt::tt_metal::CoreCoord(core.x, core.y);
}

tt::tt_metal::CoreCoord get_router_virtual_core(const tt::Cluster& cluster, ChipId physical_chip_id, chan_id_t chan) {
    return cluster.get_virtual_coordinate_from_logical_coordinates(
        physical_chip_id, get_router_logical_core(cluster, physical_chip_id, chan), CoreType::ETH);
}

// Resolves each published router's key and cores. The router pass recorded the channel the router runs on and the
// direction its builder built it facing; ControlPlane gives its routing plane, its place among the direction's
// channels, and the cluster gives its cores.
//
// Example: a chip with two E links, one N link and one W link.
//
//   published, in Ethernet channel order:
//
//       ch 0          ch 1          ch 4          ch 9
//       dir   E       dir   E       dir   N       dir   W
//       plane ?       plane ?       plane ?       plane ?
//       cores ?       cores ?       cores ?       cores ?
//
//   resolved, with ControlPlane::get_routing_plane_id per channel:
//
//       E0            E1            N0            W0         <-- router_key(direction, routing_plane)
//       logical core  logical core  logical core  logical core  <-- from the cluster's soc descriptor
//       virtual core  virtual core  virtual core  virtual core
void resolve_router_identities(
    std::vector<manifest::Router>& routers,
    const ControlPlane& control_plane,
    const tt::Cluster& cluster,
    FabricNodeId node,
    ChipId physical_chip_id) {
    std::set<std::pair<eth_chan_directions, routing_plane_id_t>> keys;
    for (auto& router : routers) {
        auto& identity = router.identity;
        const chan_id_t chan = identity.eth_chan;
        identity.routing_plane = control_plane.get_routing_plane_id(node, chan);
        TT_FATAL(
            keys.emplace(identity.direction, identity.routing_plane).second,
            "Fabric manifest: {} has two routers keyed {}",
            node,
            router_key(identity.direction, identity.routing_plane));
        identity.logical_core = get_router_logical_core(cluster, physical_chip_id, chan);
        identity.virtual_core = get_router_virtual_core(cluster, physical_chip_id, chan);
    }
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
//
// Unlike sibling routers, that are on the same chip, the peer is on a different chip. ControlPlane reports that far end
// as a (chip, channel), and this turns it into the router living there.
//
// Example: a 1x4 ring of chips (a torus along the east-west axis), resolving the router on chip 0, ch 9 (W0).
//
//   Each router sits on the chip edge it faces, so W0 is on the west edge and E0 on the east edge.
//
//         +---------------------------------------------------------------------+
//         |                                                                     |
//         v                                                                     v
//    +----------------+   +----------------+   +----------------+   +----------------+
//    | +----+  +----+ |   | +----+  +----+ |   | +----+  +----+ |   | +----+  +----+ |
//    | | W0 |  | E0 | |   | | W0 |  | E0 | |   | | W0 |  | E0 | |   | | W0 |  | E0 | |
//    | |ch 9|  |ch 0|-------|ch 9|  |ch 0|-------|ch 9|  |ch 0|-------|ch 9|  |ch 0| |
//    | +----+  +----+ |   | +----+  +----+ |   | +----+  +----+ |   | +----+  +----+ |
//    |     chip 0     |   |     chip 1     |   |     chip 2     |   |     chip 3     |
//    +----------------+   +----------------+   +----------------+   +----------------+
//
//   W0 on chip 0 is the router being resolved. Its cable runs around the ring to E0 on chip 3. The other cables
//   are ordinary links between neighbours, for comparison.
//
//   1. peer:        ControlPlane says chip 0 ch 9 is cabled to (chip 3, ch 0). That channel is an active router
//                   facing E on plane 0, so link.peer = chip 3 E0. If the channel was trimmed from chip 3's
//                   routing planes there is no router there, and link.peer stays empty.
//   2. cross_host:  true when the cable leaves this host (the peer chip is owned by another host's process).
//   3. wrap:        true when the link closes the ring. Chip 0 and chip 3 sit at coordinates 0 and 3 along the
//                   east-west axis, and their delta (3) is the mesh length (4) minus one, so the link is a wrap.
//                   The chip 0 to chip 1 link has delta 1, so it is not.
//
//   result:  link = { peer = chip 3 E0, cross_host = false, wrap = true }
void resolve_link_peer(
    manifest::Router& router,
    const ControlPlane& control_plane,
    const std::optional<manifest::MeshTorus>& torus,
    FabricNodeId node,
    ChipId physical_chip_id) {
    const chan_id_t chan = router.identity.eth_chan;
    const auto peer = control_plane.try_get_connected_mesh_chip_chan_ids(node, chan);
    if (peer.has_value()) {
        router.link.peer = find_peer_router(control_plane, peer->first, peer->second);
    }
    router.link.cross_host = control_plane.is_cross_host_eth_link(physical_chip_id, chan);
    router.link.wrap = is_wrap_link(torus, control_plane.get_mesh_graph(), node, router.identity.direction, peer);
}

// Resolves which sibling feeds each of `router`'s sender channels: the one whose downstream edge lands on the channel,
// excluding edges that end at this router's tensix mux. An edge landing past the router's senders on its VC feeds none.
void resolve_sibling_producers(manifest::Router& router, const std::vector<manifest::Router>& routers) {
    for (const auto& sibling : routers) {
        if (&sibling == &router) {
            continue;
        }
        for (uint32_t vc = 0; vc < sibling.intra_chip_downstream_edges.size(); ++vc) {
            for (const auto& edge : sibling.intra_chip_downstream_edges[vc]) {
                auto& senders = router.channels.senders.at(vc);
                if (edge.target_eth_chan == router.identity.eth_chan && !edge.through_tensix_mux &&
                    edge.landing_channel < senders.size()) {
                    senders[edge.landing_channel].producer =
                        manifest::SiblingRouterRef{.direction = sibling.identity.direction};
                }
            }
        }
    }
}

// Resolves the Ethernet channels the builders recorded to the routers they name: each router's key and cores, the
// local sync master, each link's peer, and the sibling feeding each sender channel.
manifest::Chip resolve_chip_routers(
    manifest::Chip chip,
    const ControlPlane& control_plane,
    const tt::Cluster& cluster,
    const std::optional<manifest::MeshTorus>& torus,
    FabricNodeId node,
    ChipId physical_chip_id) {
    // Get each router's routing plane and cores
    resolve_router_identities(chip.routers, control_plane, cluster, node, physical_chip_id);

    if (chip.local_sync.has_value()) {
        chip.local_sync->master = get_master_router(chip.routers, node, chip.local_sync->master_eth_chan);
    }

    // Resolve each router's link to its off-chip peer router (sender -> receiver)
    for (auto& router : chip.routers) {
        resolve_link_peer(router, control_plane, torus, node, physical_chip_id);
    }

    // Each router resolves only its own sender channels' producers, from its siblings' edges.
    for (auto& router : chip.routers) {
        resolve_sibling_producers(router, chip.routers);
    }

    return chip;
}

manifest::Chip describe_chip(
    const ControlPlane& control_plane,
    const tt::Cluster& cluster,
    const FabricBuilderContext& builder_context,
    const std::optional<manifest::MeshTorus>& torus,
    FabricNodeId node) {
    // Get the chip's coordinates in its mesh
    const auto& mesh_graph = control_plane.get_mesh_graph();
    const MeshCoordinate mesh_coord = mesh_graph.chip_to_coordinate(node.mesh_id, node.chip_id);
    std::vector<uint32_t> coords(mesh_coord.coords().begin(), mesh_coord.coords().end());
    std::optional<uint32_t> host_rank;
    if (const auto owner = mesh_graph.get_host_rank_for_chip(node.mesh_id, node.chip_id); owner.has_value()) {
        host_rank = **owner;
    }

    // Non-local chips, where locality is at the host rank granularity, can only be described by their place in the
    // mesh. The chip's owning rank takes care of more in depth description.
    const auto physical_chip_id = control_plane.try_get_physical_chip_id_from_fabric_node_id(node);
    if (!physical_chip_id.has_value()) {
        return manifest::Chip{.fabric_node_id = node, .mesh_coord = std::move(coords), .host_rank = host_rank};
    }

    // A chip that built no routers published nothing, so it starts with no routers
    const auto& published_chips = builder_context.get_manifest_chips();
    manifest::Chip chip =
        published_chips.has(*physical_chip_id) ? published_chips.get(*physical_chip_id) : manifest::Chip{};

    // Fill in the chip's details
    chip.fabric_node_id = node;
    chip.mesh_coord = std::move(coords);
    chip.host_rank = host_rank;
    chip.physical_chip_id = physical_chip_id;
    chip.asic_id = *control_plane.get_asic_id_from_fabric_node_id(node);

    // Resolve the chip's routers against the control plane and the chip's other routers
    return resolve_chip_routers(std::move(chip), control_plane, cluster, torus, node, *physical_chip_id);
}

manifest::Mesh describe_mesh(
    const ControlPlane& control_plane,
    const tt::Cluster& cluster,
    const FabricBuilderContext& builder_context,
    MeshId mesh_id) {
    // Get the mesh's shape and configuration
    const auto& mesh_graph = control_plane.get_mesh_graph();
    const MeshShape mesh_shape = mesh_graph.get_mesh_shape(mesh_id);
    manifest::Mesh mesh{.mesh_id = mesh_id, .express_routing = control_plane.express_routing_enabled(mesh_id)};
    for (size_t dim = 0; dim < mesh_shape.dims(); ++dim) {
        mesh.shape.push_back(mesh_shape[dim]);
    }

    mesh.torus = describe_mesh_torus(mesh_graph, mesh_id);

    // The builder context plans credits only for the meshes on this host
    const auto local_mesh_ids = control_plane.get_local_mesh_id_bindings();
    if (std::ranges::find(local_mesh_ids, mesh_id) != local_mesh_ids.end()) {
        mesh.credit_transport = manifest::describe_credit_transport(builder_context, mesh_id);
    }

    // Describe each chip in the mesh
    for (const auto& [_, fabric_chip_id] : mesh_graph.get_chip_ids(mesh_id)) {
        mesh.chips.push_back(
            describe_chip(control_plane, cluster, builder_context, mesh.torus, FabricNodeId(mesh_id, fabric_chip_id)));
    }

    return mesh;
}

}  // namespace

namespace manifest {

std::vector<Mesh> describe_meshes(
    const ControlPlane& control_plane, const tt::Cluster& cluster, const FabricBuilderContext& builder_context) {
    auto mesh_ids = control_plane.get_mesh_graph().get_all_mesh_ids();
    std::ranges::sort(mesh_ids, {}, [](const MeshId& mesh_id) { return *mesh_id; });
    std::vector<Mesh> meshes;
    meshes.reserve(mesh_ids.size());
    for (const auto& mesh_id : mesh_ids) {
        meshes.push_back(describe_mesh(control_plane, cluster, builder_context, mesh_id));
    }
    return meshes;
}

}  // namespace manifest

}  // namespace tt::tt_fabric
