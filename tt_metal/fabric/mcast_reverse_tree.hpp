// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/experimental/fabric/mesh_graph.hpp>

#include "axis_route_topology.hpp"
#include "hostdevcommon/fabric_common.h"

namespace tt::tt_fabric {

// Source-rooted reverse trees for 2D action-map multicast: T(root) is the union over every destination of
// the canonical route from root along one axis. Where that union is an arborescence, a worker can
// encode a multicast by walking the edge list once from the leaves inward, setting a parent's output
// when its subtree holds a requested target.
//
// Topology comes from MeshGraph and canonical next hops from AxisRouteTopology, so a fixture can be
// checked without a cluster.

// One edge, oriented child -> parent, carrying the command the parent issues to reach the child.
struct McastTreeEdge {
    int child = 0;
    int parent = 0;
    RoutingDirection parent_output = RoutingDirection::NONE;
};

struct McastReverseTree {
    int root = 0;
    int axis_dim = 0;
    int axis_len = 0;
    // Ordered descendants-before-ancestors: an edge appears only after every edge in its child's
    // subtree, which is what the worker's single reverse pass requires.
    std::vector<McastTreeEdge> edges;
};

// The tree rooted at `root`, or nullopt with `failure` set when the canonical routes out of that root
// are not an arborescence. That is a topology property rather than a bad argument, so it is reported
// rather than thrown.
std::optional<McastReverseTree> build_mcast_reverse_tree(
    const MeshGraph& mesh_graph,
    MeshId mesh_id,
    const AxisRouteTopology& topo,
    int root,
    std::string* failure = nullptr);

// Mesh-wide gate: every root on the axis must yield an arborescence. Only one encoder exists, so a
// single failing root rejects the mesh.
struct ArborescenceGateResult {
    bool passed = false;
    int failing_root = -1;                // -1 when passed
    std::string failure;                  // empty when passed
    std::vector<McastReverseTree> trees;  // one per root, in root order; empty when the gate fails
};

ArborescenceGateResult run_mcast_arborescence_gate(
    const MeshGraph& mesh_graph, MeshId mesh_id, const AxisRouteTopology& topo);

// The one-hot Routing2DCodec::ACTION_* bit a hop in this direction asks a router to take.
std::uint8_t mcast_action_bit(RoutingDirection direction);

// Packed device descriptor. The 64-row bound fixes the two 6-bit row fields:
//
//   bits  0..5   child
//   bits  6..11  parent
//   bits 12..13  parent_output, the axis 2-bit code (Y: N/S/Z, X: E/W) the parent issues
//   bits 14..15  reserved, zero
inline constexpr int MCAST_TREE_MAX_AXIS_LEN = 64;

// Host spelling of the accessors that live with the device decode in fabric_common.h.
constexpr int mcast_tree_edge_child(std::uint16_t packed) { return Routing2DCodec::mcast_edge_child(packed); }
constexpr int mcast_tree_edge_parent(std::uint16_t packed) { return Routing2DCodec::mcast_edge_parent(packed); }
constexpr std::uint8_t mcast_tree_edge_output(std::uint16_t packed) {
    return Routing2DCodec::mcast_edge_output(packed);
}

// Serializes the edge list for the device, checking the axis bound, that each direction is
// representable on its axis, and descendants-before-ancestors order. A wrong order silently drops
// branches on device rather than failing, so it is rejected here.
std::optional<std::vector<std::uint16_t>> pack_mcast_reverse_tree(
    const McastReverseTree& tree, std::string* failure = nullptr);

// Host reference for the worker's encode pass: one action byte per row, holding the OR of the outputs
// that row must drive to reach every requested target. The device version differs only in using fixed
// uint32_t bitmaps rather than vectors.
//
// `targets` is indexed by row. LOCAL_DELIVER is not set here, since a row can be needed purely as a
// transit parent.
std::vector<std::uint8_t> encode_mcast_axis_actions(const McastReverseTree& tree, const std::vector<bool>& targets);

// The directions a same-mesh multicast leaves its source on, from the eth bits of the root action.
// Runs the same encoder the worker runs, so a host opening one connection per returned direction opens
// exactly the set the worker injects on.
//
// More than one is ordinary under express routing, e.g. a northward range leaving on both N and Z.
// Empty means a legal local-only range that delivers at the source alone.
std::vector<RoutingDirection> mcast_root_output_directions(
    const MeshGraph& mesh_graph,
    MeshId mesh_id,
    const AxisRouteTopology& y_topo,
    const AxisRouteTopology& x_topo,
    int root_y,
    int root_x,
    int n_hops,
    int s_hops,
    int e_hops,
    int w_hops,
    std::string* failure = nullptr);

// Writes this chip's two trees -- T(my_y) and T(my_x) -- into its 2D route table at the offsets the
// device loader reads. Unlike the mesh-identical unicast action maps, these are written per chip.
//
// This also serves as the mesh-wide gate, since every root on an axis is some chip's own root: refusing
// a non-arborescent tree here rejects exactly the meshes the full sweep would, at O(axis^2) per chip.
//
// Returns false with `failure` set on a non-arborescent root, an axis past the packing bound, or a
// shape whose hybrid layout does not fit the slot.
bool embed_mcast_reverse_trees(
    const MeshGraph& mesh_graph,
    MeshId mesh_id,
    const AxisRouteTopology& y_topo,
    const AxisRouteTopology& x_topo,
    int my_y,
    int my_x,
    std::uint8_t* route_table_2d,
    std::string* failure = nullptr);

}  // namespace tt::tt_fabric
