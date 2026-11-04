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
#include <utility>
#include <vector>

#include "tt_metal/fabric/builder/router_wiring_rules.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_describe.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_router_pass.hpp"

namespace tt::tt_fabric {

namespace {

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

using RoutersByChan = std::map<chan_id_t, const manifest::Router*>;

RoutersByChan index_routers_by_chan(const std::vector<manifest::Router>& routers) {
    RoutersByChan by_chan;
    for (const auto& router : routers) {
        by_chan.emplace(router.identity.eth_chan, &router);
    }
    return by_chan;
}

// The router the builder connected the router on `chan` to, that faces `direction`, or null when it connected it to
// none.
//
// Siblings are routers on the same chip, wired to each other inside it. A router's downstream edge only names the
// direction it forwards to (edge.target.direction); this finds which router that is.
//
// Example: E0 (on chan 0) has a downstream edge toward N, so the call is chan = 0, direction = N.
//
//                    +-----------+
//                    | N0 (ch 4) |
//                    +-----------+
//                     /         \
//          { 4, 9 }  /           \  { 0, 4 }
//                   /             \
//       +-----------+             +-----------+
//       | W0 (ch 9) |-------------| E0 (ch 0) |  <-- `chan`
//       +-----------+  { 0, 9 }   +-----------+
//
//   Only the pairs holding chan 0 are considered. The other end of each is checked against `direction`:
//
//       { 0, 4 }  other end = ch 4 = N0, faces N     N == direction (N)  ->  match, return N0
//       { 0, 9 }  other end = ch 9 = W0, faces W     W != direction (N)  ->  no match, try the next pair
const manifest::Router* find_connected_sibling(
    const std::vector<manifest::ConnectedRouters>& connected_routers,
    const RoutersByChan& routers_by_chan,
    chan_id_t chan,
    eth_chan_directions direction) {
    for (const auto& pair : connected_routers) {
        if (pair.eth_chan1 != chan && pair.eth_chan2 != chan) {
            continue;
        }
        const auto other = static_cast<chan_id_t>(pair.eth_chan1 == chan ? pair.eth_chan2 : pair.eth_chan1);
        const auto it = routers_by_chan.find(other);
        if (it != routers_by_chan.end() && it->second->identity.direction == direction) {
            return it->second;
        }
    }
    return nullptr;
}

tt::tt_metal::CoreCoord get_router_logical_core(const tt::Cluster& cluster, ChipId physical_chip_id, chan_id_t chan) {
    const auto core = cluster.get_soc_desc(physical_chip_id).get_eth_core_for_channel(chan, CoordSystem::LOGICAL);
    return tt::tt_metal::CoreCoord(core.x, core.y);
}

tt::tt_metal::CoreCoord get_router_virtual_core(const tt::Cluster& cluster, ChipId physical_chip_id, chan_id_t chan) {
    return cluster.get_virtual_coordinate_from_logical_coordinates(
        physical_chip_id, get_router_logical_core(cluster, physical_chip_id, chan), CoreType::ETH);
}

// Resolves each collected router's identity from its Ethernet channel: its key (direction and routing plane, ie. E0)
// and cores. Returns the routers in the order ControlPlane lists the chip's active channels.
//
// The router hooks only know which Ethernet channel a router runs on, so each collected router starts with just
// its eth_chan. ControlPlane knows what each active channel means on the chip, and the cluster knows where it sits.
//
// Example: a chip with two E links, one N link and one W link.
//
//   collected (eth_chan only, in the order the hooks ran):
//
//       ch 0          ch 1          ch 4          ch 9
//       dir   ?       dir   ?       dir   ?       dir   ?
//       plane ?       plane ?       plane ?       plane ?
//       cores ?       cores ?       cores ?       cores ?
//
//   ControlPlane::get_active_fabric_eth_channels(node), in its order, with get_routing_plane_id per channel:
//
//       (ch 9, W)    (ch 0, E)    (ch 1, E)    (ch 4, N)
//        plane 0      plane 0      plane 1      plane 0
//
//   Each active channel claims its collected router and fills in the identity, so the result is reordered to match:
//
//       ch 9          ch 0          ch 1          ch 4
//       W0            E0            E1            N0         <-- router_key(direction, routing_plane)
//       logical core  logical core  logical core  logical core  <-- from the cluster's soc descriptor
//       virtual core  virtual core  virtual core  virtual core
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

// Resolves whether each of `router`'s downstream edges ends at the tensix mux of the sibling it lands on.
//
// A downstream edge names a sibling and a sender channel on it, but a sibling built with the tensix extension does not
// take every VC straight into its erisc router: VC0 goes through a tensix mux first.
//
// Example: W0 (ch 9) has one downstream edge on each of VC0 and VC1, both toward E (the east router, E0 on ch 0),
// which has the tensix extension. Routers sit on the chip edge they face, with the tensix mux in front of E0.
//
//    +---------------------------------------------------------------------------------------+
//    |                                                                                       |
//    |  +-------------+                                                +-----------------+   |
//    |  | W0 (ch 9)   |  VC0 edge        +------------+  feeds E0      | E0 (ch 0)       |   |
//    |  | (`router`)  |----------------->| tensix mux |--------------->| has tensix ext  |   |
//    |  |             |  lands on mux    +------------+                |                 |   |
//    |  |             |                                                |                 |   |
//    |  |             |  VC1 edge, lands on erisc router directly      |                 |   |
//    |  |             |----------------------------------------------->| erisc router    |   |
//    |  +-------------+                                                +-----------------+   |
//    |                                                                                       |
//    | chip                                                                                  |
//    +---------------------------------------------------------------------------------------+
void resolve_edges_through_tensix_mux(
    manifest::Router& router,
    const std::vector<manifest::ConnectedRouters>& connected_routers,
    const RoutersByChan& routers_by_chan) {
    const chan_id_t chan = router.identity.eth_chan;
    for (uint32_t vc = 0; vc < router.intra_chip_downstream_edges.size(); ++vc) {
        for (auto& edge : router.intra_chip_downstream_edges[vc]) {
            const auto* sibling =
                find_connected_sibling(connected_routers, routers_by_chan, chan, edge.target.direction);
            edge.through_tensix_mux =
                sibling != nullptr &&
                builder_type_for_vc(vc, sibling->shape.has_tensix_extension) == BuilderType::TENSIX;
        }
    }
}

// Resolves which sibling feeds each of `router`'s sender channels: the one whose downstream edge lands on the channel,
// excluding edges that end at this router's tensix mux.
void resolve_sibling_producers(
    manifest::Router& router,
    const std::vector<manifest::Router>& routers,
    const std::vector<manifest::ConnectedRouters>& connected_routers,
    const RoutersByChan& routers_by_chan) {
    for (const auto& sibling : routers) {
        if (&sibling == &router) {
            continue;
        }
        for (uint32_t vc = 0; vc < sibling.intra_chip_downstream_edges.size(); ++vc) {
            for (const auto& edge : sibling.intra_chip_downstream_edges[vc]) {
                const auto* lands_on = find_connected_sibling(
                    connected_routers, routers_by_chan, sibling.identity.eth_chan, edge.target.direction);
                auto& senders = router.channels.senders.at(vc);
                if (lands_on == &router && !edge.through_tensix_mux && edge.landing_channel < senders.size()) {
                    senders[edge.landing_channel].producer =
                        manifest::SiblingRouterRef{.direction = sibling.identity.direction};
                }
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
    // Get the router's routing plane and facing direction
    chip.routers = resolve_router_identities(std::move(chip.routers), control_plane, cluster, node, physical_chip_id);

    if (chip.local_sync.has_value()) {
        chip.local_sync->master = get_master_router(chip.routers, node, chip.local_sync->master_eth_chan);
    }

    // Resolve each router's link to its off-chip peer router (sender -> receiver)
    for (auto& router : chip.routers) {
        resolve_link_peer(router, control_plane, fabric_type, node, physical_chip_id);
    }

    // Each router resolves only its own edges, then only its own sender channels' producers, from the edges.
    const RoutersByChan routers_by_chan = index_routers_by_chan(chip.routers);
    for (auto& router : chip.routers) {
        resolve_edges_through_tensix_mux(router, chip.connected_routers, routers_by_chan);
    }
    for (auto& router : chip.routers) {
        resolve_sibling_producers(router, chip.routers, chip.connected_routers, routers_by_chan);
    }

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
    const auto& published_chips = builder_context.get_manifest_chips();
    manifest::Chip chip =
        published_chips.has(*physical_chip_id) ? published_chips.get(*physical_chip_id) : manifest::Chip{};

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
