// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/fabric/mesh_graph.hpp>

namespace tt::tt_fabric {

// Which axis a protected ring runs along. Y carries cardinal N/S plus express Z; X is the
// orthogonal cardinal E/W ring.
enum class RoutingDimension : uint8_t { X = 0, Y = 1 };

// The Y-axis topology of one mesh, projected down to a single column.
//
// The supported deployment repeats the same Y structure in every X column and the same X ring in
// every Y row, so one projection describes the whole mesh. Route generation validates that
// uniformity separately (assessment C15); this model consumes it as an input.
struct ExpressYProjection {
    uint32_t num_rows = 0;

    // Ordinary (cardinal N/S) Y edges, undirected and normalized to (lo, hi). Includes the
    // cardinal end wrap as (0, num_rows - 1) when the topology has one.
    std::vector<std::pair<uint32_t, uint32_t>> ordinary_edges;

    // Express (Z chord) Y edges, undirected and normalized to (lo, hi).
    std::vector<std::pair<uint32_t, uint32_t>> express_edges;

    bool has_cardinal_end_wrap() const;
};

// One selected physical ring family, in its canonical forward orientation.
//
// A family is a physical identity with two directed views: the forward node cycle below, and its
// reverse. Each directed view carries its own bubble on each routing plane, so "one family" is not
// "one shared bidirectional bubble".
struct ExpressRingFamily {
    // Derived express span of the family's class: bypassed ordinary hops + 1. A spanning family
    // formed when the cardinal end wrap is absent covers several classes, and takes the largest
    // span among them so the cross-family ordering rule stays well defined.
    uint32_t span = 0;

    // Canonical forward node cycle, rotated to begin at the smallest member.
    std::vector<uint32_t> forward_order;

    // Directed edge sets for the two orientations, derived from forward_order.
    std::set<std::pair<uint32_t, uint32_t>> forward_edges;
    std::set<std::pair<uint32_t, uint32_t>> reverse_edges;

    bool contains_directed(uint32_t from, uint32_t to) const;
};

// Protected-ring facts for one mesh, derived from its logical topology.
//
// This is the state behind the ControlPlane predicates that FabricBuilder consumes
// (GALAXY_BUILDER_ROUTING_CONFIG_CONTRACT.md section 4.2). Builder never sees ring identities,
// ordered cycles, or transition policy -- only the boolean answers below.
//
// Derivation is fail-closed: an unsupported or ambiguous topology throws rather than picking an
// arrangement by traversal order.
class ProtectedRingModel {
public:
    // Derive from a Y projection plus the X-ring extent. `x_ring_closed` says whether the E/W
    // edges at each row form a cycle; when false there is no protected X ring.
    static ProtectedRingModel derive(const ExpressYProjection& projection, uint32_t num_cols, bool x_ring_closed);

    // Project a mesh out of the logical graph, then derive. Throws if the mesh's Y columns are not
    // structurally identical, since the compact relation depends on that uniformity.
    static ProtectedRingModel derive_from_mesh_graph(const MeshGraph& mesh_graph, MeshId mesh_id);

    // True when this mesh has at least one materialized express (Z) adjacency. Mirrors the
    // neighbor graph by construction -- it is not an independent knob.
    bool express_enabled() const { return express_enabled_; }

    // --- Predicate surface consumed by FabricBuilder (via ControlPlane) ---

    // Does `row` (Y) or any column (X) sit on a protected ring in that dimension? A Y leaf can
    // still belong to the X ring, so this is never a per-chip bit.
    bool has_protected_ring(uint32_t row, RoutingDimension dimension) const;

    // Is the directed edge leaving `row` through `egress` a cyclic resource of some protected ring?
    bool is_protected_ring_edge(uint32_t row, RoutingDirection egress) const;

    // Do the producing hop into `row` (over the link facing `ingress`) and the hop leaving through
    // `egress` belong to the same directed ring? True means same-ring transit, so the sender needs
    // only the transit guard.
    bool are_same_directed_ring_edges(uint32_t row, RoutingDirection ingress, RoutingDirection egress) const;

    // Is a non-transit acquisition of the egress ring legal for this turn? Only meaningful once the
    // egress is known protected and the turn is not same-ring transit.
    bool continuation_allowed(uint32_t row, RoutingDirection ingress, RoutingDirection egress) const;

    // --- Diagnostics and regression accessors (not part of the builder surface) ---

    const std::vector<ExpressRingFamily>& families() const { return families_; }
    const std::set<uint32_t>& leaves() const { return leaves_; }
    // Each leaf's unique adjacent transit row.
    const std::map<uint32_t, uint32_t>& anchors() const { return anchors_; }

    // Resolve the Y row reached by leaving `row` through a Y direction, if such an edge exists.
    std::optional<uint32_t> neighbor_row(uint32_t row, RoutingDirection direction) const;

private:
    // Locate the family and orientation owning a directed Y edge.
    struct DirectedRingRef {
        size_t family_index = 0;
        bool forward = false;
    };
    std::optional<DirectedRingRef> find_directed(uint32_t from, uint32_t to) const;
    // Which family, if any, has `row` as a member. Distinguishes a cross-family crossover from an
    // off-ring leaf attachment when the connecting edge is not itself cyclic.
    std::optional<size_t> family_of_member(uint32_t row) const;

    static bool is_y_direction(RoutingDirection d);
    static bool is_x_direction(RoutingDirection d);

    uint32_t num_rows_ = 0;
    uint32_t num_cols_ = 0;
    bool x_ring_closed_ = false;
    bool express_enabled_ = false;

    ExpressYProjection projection_;
    std::vector<ExpressRingFamily> families_;
    std::set<uint32_t> transit_rows_;
    std::set<uint32_t> leaves_;
    std::map<uint32_t, uint32_t> anchors_;
    // Row -> express partner, for resolving a Z hop.
    std::map<uint32_t, uint32_t> express_partner_;
};

}  // namespace tt::tt_fabric
