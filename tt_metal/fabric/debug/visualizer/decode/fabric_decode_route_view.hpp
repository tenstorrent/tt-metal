// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The route view: what walking a packet's route needs from the manifest, built once per rank. Hops are placed by mesh
// coordinate: a router facing a direction on chip (y, x) reaches the chip one step that way, so every hop has a chip,
// on this rank or not. Only the routers are this rank's: their facing, link peer and each receiver's forwarding VC.

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <tuple>
#include <vector>

#include <hostdevcommon/fabric_common.h>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"

namespace tt::tt_fabric::decode {

// A chip's place in its mesh: its row (y) and its column (x), as the manifest's mesh_coord gives them.
struct MeshCoord {
    uint32_t y = 0;
    uint32_t x = 0;

    bool operator==(const MeshCoord&) const = default;
};

// What a walk needs of one mesh.
struct RouteMesh {
    MeshId mesh_id{0};
    uint32_t num_rows = 0;
    uint32_t num_columns = 0;
    // Which axes wrap: a step past the edge of a wrapping axis comes in at the other edge.
    manifest::MeshTorus torus;
    // The axis the express links run along. Null when express routing is off.
    std::optional<Fabric2DAxis> express_axis;
    // Whether the corner chip, at (0, 0), has exactly two neighbors. In 1D the fabric then wires the two routers of
    // every chip with exactly two neighbors together, so a forward turns the corner
    // (FabricContext::is_wrap_around_mesh, FabricBuilder::get_router_connection_pairs).
    bool wraps_around = false;
    // The chip at each coordinate, row by row. Null where the manifest lists no chip.
    std::vector<std::optional<LogicalChipId>> chip_by_coord;
};

// What a walk needs of one receiver channel. Every ERISC that runs the channel is fed it alike: since the router
// rewrite each ERISC runs one role, so ERISCs that disagree are a manifest bug.
struct RouteReceiver {
    // The VC whose downstreams the channel forwards on. Null when it forwards to no sibling, or no ERISC runs it.
    std::optional<uint32_t> forwards_on;
    // Whether its packets arrive from another mesh, so the router reads their map Y first (its intermesh_ingress
    // field). Always false in 1D.
    bool intermesh_ingress = false;
};

// What a walk needs of one router on this rank.
struct RouteRouter {
    manifest::RouterRef ref;
    // The direction the router faces.
    eth_chan_directions facing = eth_chan_directions::EAST;
    // The router on the other end of its link. Null when it has none.
    std::optional<manifest::RouterRef> peer;
    // Whether its link crosses a mesh boundary.
    bool is_intermesh = false;
    // Its receiver channels, [vc][channel].
    std::vector<std::vector<RouteReceiver>> receivers;
};

// The direction the router's kernel faces: its my_direction field, which every ERISC of a router is fed alike. Fails
// when the router's manifest entry has none.
eth_chan_directions find_router_facing(const manifest::Router& router);

// The direction opposite `direction`: north and south, east and west. Z is its own opposite: a Z link joins two Z
// routers.
eth_chan_directions get_opposite_direction(eth_chan_directions direction);

// The coordinate one step `direction` from `coord` in `mesh`: north is y - 1, south y + 1, east x + 1, west x - 1,
// wrapping on a torus axis. Null past an edge that doesn't wrap, along an axis of one chip, and for Z, which isn't a
// coordinate step.
std::optional<MeshCoord> step_mesh_coord(const RouteMesh& mesh, MeshCoord coord, eth_chan_directions direction);

class RouteView {
public:
    // Every mesh and chip the manifest lists, and every router on this rank. Fails on a manifest that contradicts
    // itself: a mesh that isn't 2D, a chip off its mesh or listed twice, a router with no my_direction, two routers of
    // one chip and routing plane with the same facing, or a receiver whose ERISCs forward on different VCs.
    explicit RouteView(const manifest::ManifestFacts& facts);

    // The mesh by its id, or null when the manifest has none.
    const RouteMesh* find_mesh(MeshId mesh_id) const;
    // Where a chip is in its mesh, or null when the manifest lists no such chip.
    std::optional<MeshCoord> find_chip_coord(FabricNodeId chip) const;
    // The chip at `coord` of mesh `mesh_id`, or null when there's none.
    std::optional<FabricNodeId> find_chip_at(MeshId mesh_id, MeshCoord coord) const;
    // The chip one coordinate step `direction` from `chip` (step_mesh_coord), or null when there's none.
    std::optional<FabricNodeId> find_next_chip(FabricNodeId chip, eth_chan_directions direction) const;
    // The cardinal directions from `chip` that reach a chip of its mesh, one bit per eth_chan_directions (as
    // Routing2DCodec's action bits are). Zero for a chip the manifest doesn't list.
    uint8_t find_neighbor_directions(FabricNodeId chip) const;

    // The router `ref` names, or null when it isn't on this rank.
    const RouteRouter* find_router(const manifest::RouterRef& ref) const;
    // The router on `chip`, on routing plane `routing_plane`, that faces `facing`: the one whose receivers take what
    // arrives from that side. Null when it isn't on this rank.
    const RouteRouter* find_receiving_router(
        FabricNodeId chip, routing_plane_id_t routing_plane, eth_chan_directions facing) const;

private:
    std::vector<RouteMesh> meshes_;
    std::map<FabricNodeId, MeshCoord> coord_by_chip_;
    std::vector<RouteRouter> routers_;
    // Indices into routers_, by key (chip, key direction, routing plane) and by facing (chip, routing plane, facing).
    std::map<std::tuple<FabricNodeId, eth_chan_directions, routing_plane_id_t>, size_t> router_by_key_;
    std::map<std::tuple<FabricNodeId, routing_plane_id_t, eth_chan_directions>, size_t> router_by_facing_;
};

}  // namespace tt::tt_fabric::decode
