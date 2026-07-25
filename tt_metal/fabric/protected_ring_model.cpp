// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/protected_ring_model.hpp"

#include <algorithm>
#include <deque>
#include <functional>
#include <unordered_map>

#include <tt_stl/assert.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_fabric {

namespace {

using EdgeSet = std::set<std::pair<uint32_t, uint32_t>>;
using Adjacency = std::map<uint32_t, std::set<uint32_t>>;

std::pair<uint32_t, uint32_t> normalized(uint32_t a, uint32_t b) {
    return a <= b ? std::pair{a, b} : std::pair{b, a};
}

Adjacency build_adjacency(const std::vector<std::pair<uint32_t, uint32_t>>& edges) {
    Adjacency adj;
    for (const auto& [a, b] : edges) {
        adj[a].insert(b);
        adj[b].insert(a);
    }
    return adj;
}

struct OrdinaryWalk {
    std::vector<uint32_t> path;  // inclusive of both endpoints
    bool ambiguous = false;      // two or more distinct shortest walks
    bool found = false;
};

// Shortest walk between two rows over ordinary Y edges only. Ties are reported rather than broken:
// a tie means the express edge's class cannot be attributed to one bypassed span, which the initial
// contract treats as unsupported (CP contract section 4.2, rule 5).
OrdinaryWalk shortest_ordinary_walk(const Adjacency& ordinary, uint32_t from, uint32_t to) {
    OrdinaryWalk result;
    if (from == to) {
        result.found = true;
        result.path = {from};
        return result;
    }

    std::map<uint32_t, uint32_t> dist;
    std::map<uint32_t, uint32_t> path_count;
    std::map<uint32_t, uint32_t> parent;
    std::deque<uint32_t> queue{from};
    dist[from] = 0;
    path_count[from] = 1;

    while (!queue.empty()) {
        const uint32_t node = queue.front();
        queue.pop_front();
        auto it = ordinary.find(node);
        if (it == ordinary.end()) {
            continue;
        }
        for (const uint32_t next : it->second) {
            if (!dist.contains(next)) {
                dist[next] = dist[node] + 1;
                path_count[next] = path_count[node];
                parent[next] = node;
                queue.push_back(next);
            } else if (dist[next] == dist[node] + 1) {
                // Another equally short way in.
                path_count[next] += path_count[node];
            }
        }
    }

    if (!dist.contains(to)) {
        return result;
    }
    result.found = true;
    result.ambiguous = path_count[to] > 1;

    for (uint32_t node = to;; node = parent.at(node)) {
        result.path.push_back(node);
        if (node == from) {
            break;
        }
    }
    std::reverse(result.path.begin(), result.path.end());
    return result;
}

// Walk a degree-2 edge set starting at `start`, stepping to `first_step` first. Returns the node
// order if the walk is a single cycle covering every member, otherwise nullopt.
std::optional<std::vector<uint32_t>> walk_cycle(
    const Adjacency& adj, const std::set<uint32_t>& members, uint32_t start, uint32_t first_step) {
    std::vector<uint32_t> order{start};
    uint32_t previous = start;
    uint32_t current = first_step;

    while (current != start) {
        order.push_back(current);
        if (order.size() > members.size()) {
            return std::nullopt;
        }
        auto it = adj.find(current);
        if (it == adj.end() || it->second.size() != 2) {
            return std::nullopt;
        }
        uint32_t next = 0;
        bool advanced = false;
        for (const uint32_t candidate : it->second) {
            if (candidate != previous) {
                next = candidate;
                advanced = true;
            }
        }
        if (!advanced) {
            return std::nullopt;
        }
        previous = current;
        current = next;
    }

    if (order.size() != members.size()) {
        return std::nullopt;  // closed early: several disjoint cycles
    }
    return order;
}

// Rotate to begin at the smallest member and pick the lexicographically smaller of the two
// orientations as canonical forward (CP contract section 4.5).
std::vector<uint32_t> canonical_forward(const Adjacency& adj, const std::set<uint32_t>& members) {
    const uint32_t smallest = *members.begin();
    const auto& neighbors = adj.at(smallest);
    TT_FATAL(neighbors.size() == 2, "Protected ring member {} has selected degree {}", smallest, neighbors.size());

    // The two walks out of the smallest member are the family's two orientations.
    std::vector<std::vector<uint32_t>> walks;
    walks.reserve(2);
    for (const uint32_t first_step : neighbors) {
        auto walk = walk_cycle(adj, members, smallest, first_step);
        TT_FATAL(walk.has_value(), "Selected protected-ring edges do not form a single cycle covering its members");
        walks.push_back(std::move(*walk));
    }
    TT_FATAL(walks.size() == 2, "Expected two orientations for a protected ring family, got {}", walks.size());
    return std::min(walks[0], walks[1]);
}

// The X ring is closed when a row's E/W edges form a cycle over every column. Under the uniform
// four-chip X ring of the supported deployment that reduces to the wrap edge being present, so this
// looks for an E/W neighbour of column 0 sitting in the last column.
bool x_ring_is_closed(
    const MeshGraph& mesh_graph,
    MeshId mesh_id,
    uint32_t num_cols,
    const std::vector<std::unordered_map<ChipId, RouterEdge>>& mesh_edges) {
    if (num_cols <= 2) {
        return false;  // a one- or two-column axis has no distinct wrap edge
    }
    const ChipId first = mesh_graph.coordinate_to_chip(mesh_id, MeshCoordinate(0, 0));
    if (first < 0 || static_cast<size_t>(first) >= mesh_edges.size()) {
        return false;
    }
    for (const auto& [neighbor_chip, edge] : mesh_edges[first]) {
        if (edge.port_direction != RoutingDirection::E && edge.port_direction != RoutingDirection::W) {
            continue;
        }
        if (mesh_graph.chip_to_coordinate(mesh_id, neighbor_chip)[1] == num_cols - 1) {
            return true;
        }
    }
    return false;
}

}  // namespace

bool ExpressYProjection::has_cardinal_end_wrap() const {
    if (num_rows < 3) {
        return false;  // a 2-row axis has no distinct wrap edge
    }
    const auto wrap = normalized(0, num_rows - 1);
    return std::find(ordinary_edges.begin(), ordinary_edges.end(), wrap) != ordinary_edges.end();
}

bool ExpressRingFamily::contains_directed(uint32_t from, uint32_t to) const {
    const std::pair<uint32_t, uint32_t> edge{from, to};
    return forward_edges.contains(edge) || reverse_edges.contains(edge);
}

bool ProtectedRingModel::is_y_direction(RoutingDirection d) {
    return d == RoutingDirection::N || d == RoutingDirection::S || d == RoutingDirection::Z;
}

bool ProtectedRingModel::is_x_direction(RoutingDirection d) {
    return d == RoutingDirection::E || d == RoutingDirection::W;
}

ProtectedRingModel ProtectedRingModel::derive(
    const ExpressYProjection& projection, uint32_t num_cols, bool x_ring_closed) {
    ProtectedRingModel model;
    model.num_rows_ = projection.num_rows;
    model.num_cols_ = num_cols;
    model.x_ring_closed_ = x_ring_closed;
    model.projection_ = projection;
    model.express_enabled_ = !projection.express_edges.empty();

    // Endpoint order is a documented precondition, but normalize anyway: a swapped pair would still
    // build a correct adjacency while silently failing the by-value edge lookups below.
    for (auto& edge : model.projection_.ordinary_edges) {
        edge = normalized(edge.first, edge.second);
    }
    for (auto& edge : model.projection_.express_edges) {
        edge = normalized(edge.first, edge.second);
    }
    // Everything below derives from the normalized copy, which is also what the predicates read.
    const ExpressYProjection& norm = model.projection_;

    if (!model.express_enabled_) {
        // No express topology: this model contributes no Y ring. The X ring, if closed, still
        // applies, and legacy cardinal configurations keep their existing behaviour.
        return model;
    }

    const Adjacency ordinary = build_adjacency(norm.ordinary_edges);

    // Each row terminates at most one express chord; more than one is rejected by the initial
    // contract rather than compressed into a bare Z command.
    for (const auto& [a, b] : norm.express_edges) {
        TT_FATAL(
            !model.express_partner_.contains(a) && !model.express_partner_.contains(b),
            "Row {} or {} has more than one express (Z) neighbour, which the initial express contract rejects",
            a,
            b);
        model.express_partner_[a] = b;
        model.express_partner_[b] = a;
    }

    // --- Express classes by (axis, span). Span is bypassed ordinary hops + 1. ---
    std::map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>> edges_by_span;
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> bypassed_interior;
    for (const auto& edge : norm.express_edges) {
        const auto walk = shortest_ordinary_walk(ordinary, edge.first, edge.second);
        TT_FATAL(
            walk.found,
            "Express edge {}<->{} has no ordinary Y walk between its endpoints, so its class cannot be derived",
            edge.first,
            edge.second);
        TT_FATAL(
            !walk.ambiguous,
            "Express edge {}<->{} has two equally short ordinary Y walks, so its span is ambiguous",
            edge.first,
            edge.second);
        const uint32_t span = static_cast<uint32_t>(walk.path.size() - 1) + 1;
        edges_by_span[span].push_back(edge);
        bypassed_interior[edge] = std::vector<uint32_t>(walk.path.begin() + 1, walk.path.end() - 1);
    }

    // --- Leaves: interiors bypassed by the minimum-span class that terminate no express edge. ---
    const uint32_t min_span = edges_by_span.begin()->first;
    for (const auto& edge : edges_by_span.at(min_span)) {
        for (const uint32_t interior : bypassed_interior.at(edge)) {
            if (!model.express_partner_.contains(interior)) {
                model.leaves_.insert(interior);
            }
        }
    }

    for (uint32_t row = 0; row < norm.num_rows; ++row) {
        if (!model.leaves_.contains(row)) {
            model.transit_rows_.insert(row);
        }
    }

    // Each leaf must pair with exactly one other leaf and have exactly one transit neighbour, its
    // anchor. Anything else means the topology is outside the supported endpoint-only shape.
    for (const uint32_t leaf : model.leaves_) {
        auto it = ordinary.find(leaf);
        TT_FATAL(it != ordinary.end(), "Leaf row {} has no ordinary Y edge", leaf);
        std::vector<uint32_t> leaf_neighbors;
        std::vector<uint32_t> transit_neighbors;
        for (const uint32_t neighbor : it->second) {
            (model.leaves_.contains(neighbor) ? leaf_neighbors : transit_neighbors).push_back(neighbor);
        }
        TT_FATAL(
            leaf_neighbors.size() == 1 && transit_neighbors.size() == 1,
            "Leaf row {} must have exactly one paired leaf and one anchor, found {} leaf and {} transit neighbours",
            leaf,
            leaf_neighbors.size(),
            transit_neighbors.size());
        model.anchors_[leaf] = transit_neighbors.front();
    }

    // --- Ring families. ---
    // With the cardinal end wrap present, each express class closes its own family. Without it, the
    // contract requires one system-spanning family over the union of classes, because no per-class
    // cycle can close.
    std::vector<std::pair<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>>> family_inputs;
    if (norm.has_cardinal_end_wrap()) {
        for (const auto& [span, edges] : edges_by_span) {
            family_inputs.emplace_back(span, edges);
        }
    } else {
        std::vector<std::pair<uint32_t, uint32_t>> all_express;
        for (const auto& [span, edges] : edges_by_span) {
            all_express.insert(all_express.end(), edges.begin(), edges.end());
        }
        family_inputs.emplace_back(edges_by_span.rbegin()->first, std::move(all_express));
    }

    for (const auto& [span, express_edges] : family_inputs) {
        std::set<uint32_t> members;
        if (family_inputs.size() == 1) {
            members = model.transit_rows_;
        } else {
            for (const auto& [a, b] : express_edges) {
                members.insert(a);
                members.insert(b);
            }
        }

        // Every member needs selected degree two. Its express chord supplies at most one, so the
        // rest must come from ordinary edges between members.
        std::map<uint32_t, uint32_t> needed;
        for (const uint32_t member : members) {
            uint32_t from_express = 0;
            auto partner = model.express_partner_.find(member);
            if (partner != model.express_partner_.end() && members.contains(partner->second)) {
                const auto edge = normalized(member, partner->second);
                if (std::find(express_edges.begin(), express_edges.end(), edge) != express_edges.end()) {
                    from_express = 1;
                }
            }
            TT_FATAL(
                from_express <= 2,
                "Row {} would need selected degree above two in a protected ring family",
                member);
            needed[member] = 2 - from_express;
        }

        std::vector<std::pair<uint32_t, uint32_t>> candidates;
        for (const auto& edge : norm.ordinary_edges) {
            if (members.contains(edge.first) && members.contains(edge.second)) {
                candidates.push_back(edge);
            }
        }
        std::sort(candidates.begin(), candidates.end());

        // Deterministic search over ordinary-edge selections that satisfy the degree requirement.
        // Collect up to two solutions so an ambiguous arrangement fails closed rather than being
        // resolved by traversal order.
        std::vector<EdgeSet> solutions;
        std::vector<bool> chosen(candidates.size(), false);
        uint64_t steps = 0;
        constexpr uint64_t k_max_steps = 2'000'000;

        auto remaining_capacity = [&](size_t from_index, uint32_t row) {
            uint32_t capacity = 0;
            for (size_t i = from_index; i < candidates.size(); ++i) {
                if (candidates[i].first == row || candidates[i].second == row) {
                    ++capacity;
                }
            }
            return capacity;
        };

        std::function<void(size_t, std::map<uint32_t, uint32_t>&)> search = [&](size_t index,
                                                                              std::map<uint32_t, uint32_t>& deficit) {
            TT_FATAL(++steps < k_max_steps, "Protected-ring arrangement search did not converge; topology unsupported");
            if (solutions.size() > 1) {
                return;  // already ambiguous
            }
            if (index == candidates.size()) {
                for (const auto& [row, remaining] : deficit) {
                    if (remaining != 0) {
                        return;
                    }
                }
                EdgeSet selected;
                for (size_t i = 0; i < candidates.size(); ++i) {
                    if (chosen[i]) {
                        selected.insert(candidates[i]);
                    }
                }
                solutions.push_back(std::move(selected));
                return;
            }

            // Prune: any row whose deficit exceeds what the remaining candidates can supply.
            for (const auto& [row, remaining] : deficit) {
                if (remaining > remaining_capacity(index, row)) {
                    return;
                }
            }

            const auto& [a, b] = candidates[index];
            if (deficit[a] > 0 && deficit[b] > 0) {
                --deficit[a];
                --deficit[b];
                chosen[index] = true;
                search(index + 1, deficit);
                chosen[index] = false;
                ++deficit[a];
                ++deficit[b];
            }
            search(index + 1, deficit);
        };

        std::map<uint32_t, uint32_t> deficit = needed;
        search(0, deficit);

        TT_FATAL(
            !solutions.empty(),
            "No valid protected-ring arrangement exists for the span-{} family over {} rows",
            span,
            members.size());
        TT_FATAL(
            solutions.size() == 1,
            "More than one distinct protected-ring arrangement exists for the span-{} family; the initial contract "
            "has no preference among them, so this topology is unsupported",
            span);

        std::vector<std::pair<uint32_t, uint32_t>> selected_edges(express_edges.begin(), express_edges.end());
        selected_edges.insert(selected_edges.end(), solutions.front().begin(), solutions.front().end());
        const Adjacency selected_adj = build_adjacency(selected_edges);

        ExpressRingFamily family;
        family.span = span;
        family.forward_order = canonical_forward(selected_adj, members);
        const size_t count = family.forward_order.size();
        for (size_t i = 0; i < count; ++i) {
            const uint32_t from = family.forward_order[i];
            const uint32_t to = family.forward_order[(i + 1) % count];
            family.forward_edges.insert({from, to});
            family.reverse_edges.insert({to, from});
        }
        model.families_.push_back(std::move(family));
    }

    // The families must partition the transit rows exactly: no transit row left out, none shared.
    std::set<uint32_t> covered;
    for (const auto& family : model.families_) {
        for (const uint32_t row : family.forward_order) {
            TT_FATAL(covered.insert(row).second, "Row {} belongs to more than one protected ring family", row);
        }
    }
    TT_FATAL(
        covered == model.transit_rows_,
        "Protected ring families cover {} rows but there are {} transit rows",
        covered.size(),
        model.transit_rows_.size());

    return model;
}

std::optional<uint32_t> ProtectedRingModel::neighbor_row(uint32_t row, RoutingDirection direction) const {
    if (direction == RoutingDirection::Z) {
        auto it = express_partner_.find(row);
        return it == express_partner_.end() ? std::nullopt : std::optional{it->second};
    }
    if (direction != RoutingDirection::N && direction != RoutingDirection::S) {
        return std::nullopt;
    }
    if (num_rows_ == 0) {
        return std::nullopt;
    }

    // N decreases logical Y, S increases it; both include the cardinal wrap when it exists.
    uint32_t candidate = 0;
    if (direction == RoutingDirection::N) {
        candidate = row == 0 ? num_rows_ - 1 : row - 1;
    } else {
        candidate = row + 1 == num_rows_ ? 0 : row + 1;
    }

    const auto edge = normalized(row, candidate);
    const auto& edges = projection_.ordinary_edges;
    if (std::find(edges.begin(), edges.end(), edge) == edges.end()) {
        return std::nullopt;
    }
    return candidate;
}

std::optional<size_t> ProtectedRingModel::family_of_member(uint32_t row) const {
    for (size_t i = 0; i < families_.size(); ++i) {
        const auto& order = families_[i].forward_order;
        if (std::find(order.begin(), order.end(), row) != order.end()) {
            return i;
        }
    }
    return std::nullopt;
}

std::optional<ProtectedRingModel::DirectedRingRef> ProtectedRingModel::find_directed(uint32_t from, uint32_t to) const {
    for (size_t i = 0; i < families_.size(); ++i) {
        if (families_[i].forward_edges.contains({from, to})) {
            return DirectedRingRef{i, true};
        }
        if (families_[i].reverse_edges.contains({from, to})) {
            return DirectedRingRef{i, false};
        }
    }
    return std::nullopt;
}

bool ProtectedRingModel::has_protected_ring(uint32_t row, RoutingDimension dimension) const {
    if (dimension == RoutingDimension::X) {
        return x_ring_closed_;
    }
    for (const auto& family : families_) {
        if (std::find(family.forward_order.begin(), family.forward_order.end(), row) != family.forward_order.end()) {
            return true;
        }
    }
    return false;
}

bool ProtectedRingModel::is_protected_ring_edge(uint32_t row, RoutingDirection egress) const {
    if (is_x_direction(egress)) {
        return x_ring_closed_;
    }
    if (!is_y_direction(egress)) {
        return false;
    }
    const auto neighbor = neighbor_row(row, egress);
    return neighbor.has_value() && find_directed(row, *neighbor).has_value();
}

bool ProtectedRingModel::are_same_directed_ring_edges(
    uint32_t row, RoutingDirection ingress, RoutingDirection egress) const {
    // The ingress direction names the local port facing the producer, so the producing hop ran from
    // that neighbour into this row.
    if (is_x_direction(ingress) && is_x_direction(egress)) {
        if (!x_ring_closed_) {
            return false;
        }
        // Forward X is increasing (E). Arriving on the W-facing port and leaving E stays forward;
        // arriving E and leaving W stays reverse. Anything else reverses orientation.
        return (ingress == RoutingDirection::W && egress == RoutingDirection::E) ||
               (ingress == RoutingDirection::E && egress == RoutingDirection::W);
    }
    if (!is_y_direction(ingress) || !is_y_direction(egress)) {
        return false;  // a dimension change is never same-ring transit
    }

    const auto producer = neighbor_row(row, ingress);
    const auto next = neighbor_row(row, egress);
    if (!producer.has_value() || !next.has_value()) {
        return false;
    }
    const auto in_ring = find_directed(*producer, row);
    const auto out_ring = find_directed(row, *next);
    if (!in_ring.has_value() || !out_ring.has_value()) {
        return false;
    }
    return in_ring->family_index == out_ring->family_index && in_ring->forward == out_ring->forward;
}

bool ProtectedRingModel::continuation_allowed(
    uint32_t row, RoutingDirection ingress, RoutingDirection egress) const {
    // Reached only once the egress is known protected and the turn is not same-ring transit, so the
    // remaining same-dimension cases are an off-ring entry, a legal cross-family acquisition, or an
    // orientation reversal.
    if (is_x_direction(ingress) && is_x_direction(egress)) {
        return false;  // same-dimension X that is not same-orientation transit is a reversal
    }
    if (!is_y_direction(ingress) || !is_y_direction(egress)) {
        return false;
    }

    const auto producer = neighbor_row(row, ingress);
    const auto next = neighbor_row(row, egress);
    if (!next.has_value()) {
        return false;
    }
    if (!producer.has_value()) {
        return true;  // no ordinary ingress edge: treat as a source-style entry
    }

    const auto in_ring = find_directed(*producer, row);
    const auto out_ring = find_directed(row, *next);
    if (!out_ring.has_value()) {
        return false;
    }

    // Larger-span family into smaller-span family may continue; the reverse lands only, so it is
    // terminal in Y rather than a continuation (CP contract section 4.8).
    const uint32_t egress_span = families_[out_ring->family_index].span;

    if (in_ring.has_value()) {
        if (in_ring->family_index == out_ring->family_index) {
            return false;  // same family, opposite orientation: an adaptive reversal, never canonical
        }
        return families_[in_ring->family_index].span > egress_span;
    }

    // The ingress edge is not itself a cyclic resource, which covers both a leaf/anchor attachment
    // and a cross-family crossover. Those are not interchangeable, so the producing row's family
    // decides: arriving from a larger-span family may continue, while arriving from a smaller-span
    // one is a terminal landing. Deciding on ingress cyclic-ness alone would wrongly admit the
    // ex4 -> ex8 landing that must stop in Y.
    const auto producer_family = family_of_member(*producer);
    if (!producer_family.has_value()) {
        return true;  // off-ring attachment, e.g. a leaf anchor: a legal first acquisition
    }
    if (*producer_family == out_ring->family_index) {
        return false;  // an edge the arrangement excluded from transit within one family
    }
    return families_[*producer_family].span > egress_span;
}

ProtectedRingModel ProtectedRingModel::derive_from_mesh_graph(const MeshGraph& mesh_graph, MeshId mesh_id) {
    const auto shape = mesh_graph.get_mesh_shape(mesh_id);
    const auto& intra = mesh_graph.get_intra_mesh_connectivity();
    const auto mesh_index = *mesh_id;
    TT_FATAL(mesh_index < intra.size(), "Mesh {} is out of range of intra-mesh connectivity", mesh_index);

    // A mesh with no same-mesh express adjacency has no Y ring to derive, and none of the structural
    // requirements below apply to it. Legacy cardinal meshes take this path: they must not be
    // rejected for a non-2D shape or for columns that differ, which an irregular carve-out can
    // legitimately have. The strict projection is an express-topology requirement, not a general one.
    bool any_express = false;
    for (const auto& chip_edges : intra[mesh_index]) {
        for (const auto& [neighbor_chip, edge] : chip_edges) {
            if (edge.port_direction == RoutingDirection::Z) {
                any_express = true;
                break;
            }
        }
        if (any_express) {
            break;
        }
    }

    if (shape.dims() != 2 || !any_express) {
        ProtectedRingModel model;
        if (shape.dims() == 2) {
            model.num_rows_ = shape[0];
            model.num_cols_ = shape[1];
            model.x_ring_closed_ = x_ring_is_closed(mesh_graph, mesh_id, shape[1], intra[mesh_index]);
        }
        return model;
    }

    const uint32_t num_rows = shape[0];
    const uint32_t num_cols = shape[1];

    ExpressYProjection projection;
    projection.num_rows = num_rows;

    EdgeSet ordinary;
    EdgeSet express;

    // Project every column, then require them to agree. The compact route relation depends on that
    // uniformity, so a mismatch is a topology this model cannot describe.
    std::optional<std::pair<EdgeSet, EdgeSet>> reference;
    for (uint32_t col = 0; col < num_cols; ++col) {
        EdgeSet column_ordinary;
        EdgeSet column_express;
        for (uint32_t row = 0; row < num_rows; ++row) {
            const ChipId chip = mesh_graph.coordinate_to_chip(mesh_id, MeshCoordinate(row, col));
            TT_FATAL(
                chip >= 0 && static_cast<size_t>(chip) < intra[mesh_index].size(),
                "Chip {} out of range in mesh {}",
                chip,
                mesh_index);
            for (const auto& [neighbor_chip, edge] : intra[mesh_index][chip]) {
                const auto neighbor_coord = mesh_graph.chip_to_coordinate(mesh_id, neighbor_chip);
                const uint32_t neighbor_row = neighbor_coord[0];
                const uint32_t neighbor_col = neighbor_coord[1];
                switch (edge.port_direction) {
                    case RoutingDirection::N:
                    case RoutingDirection::S:
                        TT_FATAL(
                            neighbor_col == col,
                            "Cardinal Y edge from chip {} leaves column {} for column {}",
                            chip,
                            col,
                            neighbor_col);
                        column_ordinary.insert(normalized(row, neighbor_row));
                        break;
                    case RoutingDirection::Z:
                        TT_FATAL(
                            neighbor_col == col,
                            "Express (Z) edge from chip {} must differ only on Y in the initial contract, but leaves "
                            "column {} for column {}",
                            chip,
                            col,
                            neighbor_col);
                        column_express.insert(normalized(row, neighbor_row));
                        break;
                    case RoutingDirection::E:
                    case RoutingDirection::W:
                        TT_FATAL(
                            neighbor_row == row,
                            "Cardinal X edge from chip {} leaves row {} for row {}",
                            chip,
                            row,
                            neighbor_row);
                        break;
                    default: break;
                }
            }
        }
        if (reference.has_value()) {
            TT_FATAL(
                reference->first == column_ordinary && reference->second == column_express,
                "Column {} has a different Y structure from column 0; the express route relation requires every "
                "column to repeat the same Y topology",
                col);
        } else {
            reference = std::pair{column_ordinary, column_express};
        }
        ordinary = column_ordinary;
        express = column_express;
    }

    projection.ordinary_edges.assign(ordinary.begin(), ordinary.end());
    projection.express_edges.assign(express.begin(), express.end());

    return derive(projection, num_cols, x_ring_is_closed(mesh_graph, mesh_id, num_cols, intra[mesh_index]));
}

}  // namespace tt::tt_fabric
