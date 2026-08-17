// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/experimental/fabric/mesh_graph.hpp>

#include "express_ring_topology.hpp"
#include "hostdevcommon/fabric_common.h"

namespace tt::tt_fabric {

// Source-rooted reverse trees for indexed multicast (codec contract §5.7.1).
//
// The tree is a representation of the canonical routes, not a new policy: T(root) is the union over
// every destination of R(root, dst) along one axis. Where that union is an arborescence, a worker can
// encode a multicast by walking the edge list once from the leaves inward, setting a parent's output
// exactly when the parent's subtree holds a requested target. That works only because each row has a
// single parent, which is what the gate below establishes.
//
// Everything here is machine-free: it reads the declared topology through MeshGraph and the canonical
// next-hop through ExpressRingTopology, so a fixture can be checked without a cluster or a matching
// host world.

// One edge, oriented child -> parent, carrying the command the parent issues to reach the child. Host
// form; the packed 16-bit device descriptor is a later artifact.
struct McastTreeEdge {
    int child = 0;
    int parent = 0;
    RoutingDirection parent_output = RoutingDirection::NONE;
};

struct McastReverseTree {
    int root = 0;
    int axis_dim = 0;
    int axis_len = 0;
    // Serialized descendants-before-ancestors, which is what makes the worker's single reverse pass
    // correct: an edge is visited only after every edge in its child's subtree.
    std::vector<McastTreeEdge> edges;
};

// The tree rooted at `root`, or nullopt with `failure` set when the canonical routes out of that root
// are not an arborescence. A failure is a property of the topology and its route policy, not a bad
// argument, so it is reported rather than thrown.
std::optional<McastReverseTree> build_mcast_reverse_tree(
    const MeshGraph& mesh_graph,
    MeshId mesh_id,
    const ExpressRingTopology& topo,
    int root,
    std::string* failure = nullptr);

// Mesh-wide gate: every root on the axis must yield an arborescence. V1 runs one encoder, so a single
// failing root rejects the mesh rather than selecting a different representation for it -- which is
// why this is checked across all roots up front instead of at encode time.
struct ArborescenceGateResult {
    bool passed = false;
    int failing_root = -1;                // -1 when passed
    std::string failure;                  // empty when passed
    std::vector<McastReverseTree> trees;  // one per root, in root order; empty when the gate fails
};

ArborescenceGateResult run_mcast_arborescence_gate(
    const MeshGraph& mesh_graph, MeshId mesh_id, const ExpressRingTopology& topo);

// The one-hot IndexedMeshRoutingFields::ACTION_* bit a hop in this direction asks a router to take.
std::uint8_t mcast_action_bit(RoutingDirection direction);

// Packed device descriptor (§5.7.1). The Y bound of 64 is what fixes the two 6-bit row fields:
//
//   bits  0..5   child
//   bits  6..11  parent
//   bits 12..13  parent_output, the axis 2-bit code (Y: N/S/Z, X: E/W) the parent issues
//   bits 14..15  reserved, zero
//
// parent_output reuses the existing IndexedMeshRoutingFields 2-bit vector encodings rather than a
// private one, so a descriptor and a unicast vector name a direction the same way.
inline constexpr int MCAST_TREE_MAX_AXIS_LEN = 64;

// Field extraction lives with the device decode in fabric_common.h; these are the host spelling of
// the same accessors, not a second copy of the layout.
constexpr int mcast_tree_edge_child(std::uint16_t packed) { return IndexedMeshRoutingFields::mcast_edge_child(packed); }
constexpr int mcast_tree_edge_parent(std::uint16_t packed) {
    return IndexedMeshRoutingFields::mcast_edge_parent(packed);
}
constexpr std::uint8_t mcast_tree_edge_output(std::uint16_t packed) {
    return IndexedMeshRoutingFields::mcast_edge_output(packed);
}

// Serializes the edge list for the device. Enforces the §5.7.1 host packing obligations: the axis
// bound, a direction that is representable on this axis, and above all descendants-before-ancestors
// order -- a wrong order does not fail loudly on device, it silently drops branches from the encoded
// map, so it is rejected here where it can still be seen.
std::optional<std::vector<std::uint16_t>> pack_mcast_reverse_tree(
    const McastReverseTree& tree, std::string* failure = nullptr);

// The §5.7.1 encode pass, host side: one action byte per row, holding the OR of the outputs that row
// must drive to reach every requested target. Walks the edge list once from the leaves inward, taking
// an edge exactly when its child subtree still holds something needed, which makes the result
// union(R(root, target)) over the requested targets.
//
// `targets` is indexed by row and excludes nothing: an entry for the root is simply never reached,
// since no edge enters the root. LOCAL_DELIVER is deliberately not set here -- §5.7.1 forbids deriving
// it from the expanded `needed` set, because a row can be needed purely as a transit parent.
//
// This is the host reference for the worker loop, and the device version differs only in using fixed
// uint32_t bitmaps rather than vectors.
std::vector<std::uint8_t> encode_mcast_axis_actions(const McastReverseTree& tree, const std::vector<bool>& targets);

// The directions a same-mesh multicast leaves its source on: the set eth bits of the canonical root
// action, as routing directions. Runs the same encoder the worker runs, over trees embedded exactly
// as the device sees them, so a host that opens a connection per returned direction opens precisely
// the set the worker will inject on (§7.3.1).
//
// Express routing makes more than one ordinary: a single northward range can leave on both N and Z.
// Returns empty when the range is local-only, which is legal and means deliver at the source alone.
std::vector<RoutingDirection> mcast_root_output_directions(
    const MeshGraph& mesh_graph,
    MeshId mesh_id,
    const ExpressRingTopology& y_topo,
    const ExpressRingTopology& x_topo,
    int root_y,
    int root_x,
    int n_hops,
    int s_hops,
    int e_hops,
    int w_hops,
    std::string* failure = nullptr);

// Writes this chip's two trees -- T(my_y) on the Y axis and T(my_x) on the X axis -- into its indexed
// vector table at the contract §6.2 offsets. The unicast vectors are mesh-identical, so they are
// generated once and copied; these are the part that differs per chip, so they are written per chip
// over the same buffer.
//
// This doubles as the §6.1 mesh-wide gate. Every root on an axis is some chip's own root, so refusing
// to embed a non-arborescent tree here rejects exactly the meshes the gate would reject, at O(axis^2)
// per chip instead of re-running the full O(axis^3) sweep for each one.
//
// Returns false with `failure` set on a non-arborescent root, an axis past the packing bound, or a
// shape whose hybrid layout does not fit the slot ([64,4], pending the §6.2 struct growth).
bool embed_mcast_reverse_trees(
    const MeshGraph& mesh_graph,
    MeshId mesh_id,
    const ExpressRingTopology& y_topo,
    const ExpressRingTopology& x_topo,
    int my_y,
    int my_x,
    std::uint8_t* table,
    std::string* failure = nullptr);

}  // namespace tt::tt_fabric
