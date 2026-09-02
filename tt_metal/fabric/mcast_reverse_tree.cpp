// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mcast_reverse_tree.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <enchantum/enchantum.hpp>
#include <fmt/format.h>

#include "hostdevcommon/fabric_common.h"

namespace tt::tt_fabric {

namespace {

constexpr int kUnset = -1;

std::string dir_name(RoutingDirection direction) { return std::string(enchantum::to_string(direction)); }

}  // namespace

std::optional<McastReverseTree> build_mcast_reverse_tree(
    const MeshGraph& mesh_graph, MeshId mesh_id, const AxisRouteTopology& topo, int root, std::string* failure) {
    const int len = topo.axis_len;
    const auto fail = [&](std::string message) -> std::optional<McastReverseTree> {
        if (failure != nullptr) {
            *failure = std::move(message);
        }
        return std::nullopt;
    };

    if (len <= 0 || root < 0 || root >= len) {
        return fail(fmt::format("root {} is out of range for an axis of length {}", root, len));
    }

    std::vector<int> parent(len, kUnset);
    std::vector<RoutingDirection> parent_output(len, RoutingDirection::NONE);

    McastReverseTree tree;
    tree.root = root;
    tree.axis_dim = topo.axis_dim;
    tree.axis_len = len;

    // Walk the canonical route to every destination and record each hop as a child->parent edge. The
    // arborescence condition is that no row is ever entered two different ways, checked here hop by hop.
    for (int dst = 0; dst < len; dst++) {
        if (dst == root) {
            continue;
        }
        int cur = root;
        for (int guard = 0; cur != dst; guard++) {
            if (guard > len) {
                return fail(fmt::format("the canonical route {} -> {} did not arrive within {} hops", root, dst, len));
            }
            const int next = topo.next_row(cur, dst);
            if (next < 0 || next >= len || next == cur) {
                return fail(fmt::format(
                    "the canonical next row from {} toward {} is {}, which is not a distinct row on this axis",
                    cur,
                    dst,
                    next));
            }
            const auto direction = axis_edge_direction(mesh_graph, mesh_id, topo.axis_dim, 0, cur, next);
            if (!direction.has_value()) {
                return fail(fmt::format(
                    "the canonical route {} -> {} takes the hop {} -> {}, but the mesh declares no edge between those "
                    "rows",
                    root,
                    dst,
                    cur,
                    next));
            }
            if (parent[next] == kUnset) {
                parent[next] = cur;
                parent_output[next] = *direction;
                tree.edges.push_back(McastTreeEdge{next, cur, *direction});
            } else if (parent[next] != cur || parent_output[next] != *direction) {
                return fail(fmt::format(
                    "row {} is entered two ways -- from {} via {}, and from {} via {} on the route to {}. T({}) is not "
                    "an arborescence, so one reverse pass cannot reproduce these routes",
                    next,
                    parent[next],
                    dir_name(parent_output[next]),
                    cur,
                    dir_name(*direction),
                    dst,
                    root));
            }
            cur = next;
        }
    }

    if (parent[root] != kUnset) {
        return fail(
            fmt::format("the root {} has a parent ({}), so it is not the source of the tree", root, parent[root]));
    }
    for (int row = 0; row < len; row++) {
        if (row != root && parent[row] == kUnset) {
            return fail(fmt::format("row {} is never entered, so it is unreachable from root {}", row, root));
        }
    }
    if (static_cast<int>(tree.edges.size()) != len - 1) {
        return fail(fmt::format(
            "T({}) has {} edges, but an arborescence over {} rows has exactly {}",
            root,
            tree.edges.size(),
            len,
            len - 1));
    }

    // Depth also serves as the acyclicity check: a row that cannot reach the root by following parents
    // is in a cycle, which the guard catches. Depth then orders the edge list.
    std::vector<int> depth(len, kUnset);
    depth[root] = 0;
    for (int row = 0; row < len; row++) {
        std::vector<int> pending;
        int cur = row;
        for (int guard = 0; depth[cur] == kUnset; guard++) {
            if (guard > len) {
                return fail(fmt::format(
                    "following parents from row {} never reaches root {}; T({}) has a cycle", row, root, root));
            }
            pending.push_back(cur);
            cur = parent[cur];
        }
        int current_depth = depth[cur];
        for (auto it = pending.rbegin(); it != pending.rend(); ++it) {
            depth[*it] = ++current_depth;
        }
    }

    // Descendants before ancestors: the worker's single pass propagates `needed` upward, so every edge
    // below a row must come before the edge above it. Ties are independent subtrees, so order is free.
    std::stable_sort(tree.edges.begin(), tree.edges.end(), [&depth](const McastTreeEdge& a, const McastTreeEdge& b) {
        return depth[a.child] > depth[b.child];
    });

    return tree;
}

std::uint8_t mcast_action_bit(RoutingDirection direction) {
    switch (direction) {
        case RoutingDirection::N: return Routing2DCodec::ACTION_NORTH;
        case RoutingDirection::S: return Routing2DCodec::ACTION_SOUTH;
        case RoutingDirection::E: return Routing2DCodec::ACTION_EAST;
        case RoutingDirection::W: return Routing2DCodec::ACTION_WEST;
        case RoutingDirection::Z: return Routing2DCodec::ACTION_Z;
        default: return 0;
    }
}

std::optional<std::vector<std::uint16_t>> pack_mcast_reverse_tree(const McastReverseTree& tree, std::string* failure) {
    const auto fail = [&](std::string message) -> std::optional<std::vector<std::uint16_t>> {
        if (failure != nullptr) {
            *failure = std::move(message);
        }
        return std::nullopt;
    };

    if (tree.axis_len > MCAST_TREE_MAX_AXIS_LEN) {
        return fail(fmt::format(
            "axis length {} exceeds the {}-row bound the 6-bit descriptor fields can address",
            tree.axis_len,
            MCAST_TREE_MAX_AXIS_LEN));
    }

    // An edge hanging off row r must come before the edge entering r, or the device pass reads r's edge
    // before r was marked needed and drops that whole branch.
    std::vector<int> position_of_edge_into(tree.axis_len, -1);
    for (std::size_t i = 0; i < tree.edges.size(); i++) {
        position_of_edge_into[tree.edges[i].child] = static_cast<int>(i);
    }

    std::vector<std::uint16_t> packed;
    packed.reserve(tree.edges.size());

    for (std::size_t i = 0; i < tree.edges.size(); i++) {
        const auto& edge = tree.edges[i];
        if (edge.child < 0 || edge.child >= tree.axis_len || edge.parent < 0 || edge.parent >= tree.axis_len) {
            return fail(fmt::format("edge {} -> {} names a row outside the axis", edge.child, edge.parent));
        }
        const int parent_edge_position = position_of_edge_into[edge.parent];
        if (parent_edge_position >= 0 && parent_edge_position < static_cast<int>(i)) {
            return fail(fmt::format(
                "edge into row {} is serialized at {}, after the edge into its parent {} at {}; the order must be "
                "descendants before ancestors",
                edge.child,
                i,
                edge.parent,
                parent_edge_position));
        }

        std::uint8_t output_code = 0;
        if (tree.axis_dim == 0) {
            switch (edge.parent_output) {
                case RoutingDirection::N: output_code = Routing2DCodec::Y2_NORTH; break;
                case RoutingDirection::S: output_code = Routing2DCodec::Y2_SOUTH; break;
                case RoutingDirection::Z: output_code = Routing2DCodec::Y2_Z; break;
                default:
                    return fail(fmt::format(
                        "edge into row {} leaves its parent {}, which the Y axis cannot encode",
                        edge.child,
                        dir_name(edge.parent_output)));
            }
        } else {
            switch (edge.parent_output) {
                case RoutingDirection::E: output_code = Routing2DCodec::X2_EAST; break;
                case RoutingDirection::W: output_code = Routing2DCodec::X2_WEST; break;
                default:
                    return fail(fmt::format(
                        "edge into row {} leaves its parent {}, which the X axis cannot encode -- X carries no express "
                        "dimension",
                        edge.child,
                        dir_name(edge.parent_output)));
            }
        }

        packed.push_back(static_cast<std::uint16_t>(
            (static_cast<std::uint16_t>(edge.child) & 0x3F) | ((static_cast<std::uint16_t>(edge.parent) & 0x3F) << 6) |
            (static_cast<std::uint16_t>(output_code) << 12)));
    }

    return packed;
}

std::vector<std::uint8_t> encode_mcast_axis_actions(const McastReverseTree& tree, const std::vector<bool>& targets) {
    std::vector<std::uint8_t> actions(tree.axis_len, 0);
    if (static_cast<int>(targets.size()) != tree.axis_len) {
        return actions;
    }

    // `needed` starts as the requested targets and grows toward the root: marking a parent needed
    // carries a target's requirement up its branch without walking the branch.
    std::vector<bool> needed = targets;
    for (const auto& edge : tree.edges) {
        if (needed[edge.child]) {
            actions[edge.parent] |= mcast_action_bit(edge.parent_output);
            needed[edge.parent] = true;
        }
    }
    return actions;
}

ArborescenceGateResult run_mcast_arborescence_gate(
    const MeshGraph& mesh_graph, MeshId mesh_id, const AxisRouteTopology& topo) {
    ArborescenceGateResult result;
    result.trees.reserve(topo.axis_len);

    for (int root = 0; root < topo.axis_len; root++) {
        std::string failure;
        auto tree = build_mcast_reverse_tree(mesh_graph, mesh_id, topo, root, &failure);
        if (!tree.has_value()) {
            result.passed = false;
            result.failing_root = root;
            result.failure = std::move(failure);
            result.trees.clear();
            return result;
        }
        result.trees.push_back(std::move(*tree));
    }

    result.passed = true;
    return result;
}

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
    std::string* failure) {
    const auto y_size = static_cast<std::uint32_t>(y_topo.axis_len);
    const auto x_size = static_cast<std::uint32_t>(x_topo.axis_len);

    // Runs the worker's encoder over a table laid out exactly as the device sees it.
    std::vector<std::uint8_t> route_table_2d(Routing2DCodec::ROUTE_TABLE_CAPACITY_BYTES, 0);
    if (!embed_mcast_reverse_trees(
            mesh_graph, mesh_id, y_topo, x_topo, root_y, root_x, route_table_2d.data(), failure)) {
        return {};
    }

    std::vector<std::uint8_t> maps(y_size + x_size, 0);
    encode_2d_mcast_maps(
        maps.data(),
        route_table_2d.data(),
        y_size,
        x_size,
        static_cast<std::uint32_t>(root_y),
        static_cast<std::uint32_t>(root_x),
        static_cast<std::uint32_t>(n_hops),
        static_cast<std::uint32_t>(s_hops),
        static_cast<std::uint32_t>(e_hops),
        static_cast<std::uint32_t>(w_hops));

    const std::uint8_t root_action = maps[root_y] & Routing2DCodec::ACTION_ETH_MASK;

    // Inverted through mcast_action_bit rather than a second bit-to-direction table.
    std::vector<RoutingDirection> directions;
    for (const auto direction :
         {RoutingDirection::E, RoutingDirection::W, RoutingDirection::N, RoutingDirection::S, RoutingDirection::Z}) {
        if ((root_action & mcast_action_bit(direction)) != 0) {
            directions.push_back(direction);
        }
    }
    return directions;
}

namespace {

// One axis of the embed: build T(root), pack it, and lay the words down at `offset`.
bool embed_one_axis(
    const MeshGraph& mesh_graph,
    MeshId mesh_id,
    const AxisRouteTopology& topo,
    int root,
    std::uint8_t* route_table_2d,
    std::uint32_t offset,
    const char* axis_label,
    std::string* failure) {
    if (root < 0 || root >= topo.axis_len) {
        if (failure != nullptr) {
            *failure = fmt::format("{} root {} is outside axis of length {}", axis_label, root, topo.axis_len);
        }
        return false;
    }

    std::string build_failure;
    auto tree = build_mcast_reverse_tree(mesh_graph, mesh_id, topo, root, &build_failure);
    if (!tree.has_value()) {
        if (failure != nullptr) {
            *failure = fmt::format("{} root {}: {}", axis_label, root, build_failure);
        }
        return false;
    }

    std::string pack_failure;
    auto packed = pack_mcast_reverse_tree(*tree, &pack_failure);
    if (!packed.has_value()) {
        if (failure != nullptr) {
            *failure = fmt::format("{} root {}: {}", axis_label, root, pack_failure);
        }
        return false;
    }

    const auto expected = Routing2DCodec::mcast_tree_edge_count(static_cast<std::uint32_t>(topo.axis_len));
    if (packed->size() != expected) {
        if (failure != nullptr) {
            *failure = fmt::format(
                "{} root {} packed {} edges, but the region holds {}", axis_label, root, packed->size(), expected);
        }
        return false;
    }

    std::uint8_t* region = route_table_2d + offset;
    for (std::uint32_t i = 0; i < expected; i++) {
        Routing2DCodec::set_mcast_tree_edge(region, i, (*packed)[i]);
    }
    return true;
}

}  // namespace

bool embed_mcast_reverse_trees(
    const MeshGraph& mesh_graph,
    MeshId mesh_id,
    const AxisRouteTopology& y_topo,
    const AxisRouteTopology& x_topo,
    int my_y,
    int my_x,
    std::uint8_t* route_table_2d,
    std::string* failure) {
    const auto y_size = static_cast<std::uint32_t>(y_topo.axis_len);
    const auto x_size = static_cast<std::uint32_t>(x_topo.axis_len);

    if (!Routing2DCodec::hybrid_region_fits(y_size, x_size)) {
        if (failure != nullptr) {
            *failure = fmt::format(
                "[{},{}] needs {} B of 2D action maps plus {} B of reverse trees, over the {} B union slot; "
                "routing_l1_info_t must grow first",
                y_size,
                x_size,
                Routing2DCodec::vectors_region_bytes(y_size, x_size),
                Routing2DCodec::mcast_tree_region_bytes(y_size, x_size),
                Routing2DCodec::ROUTE_TABLE_CAPACITY_BYTES);
        }
        return false;
    }

    return embed_one_axis(
               mesh_graph,
               mesh_id,
               y_topo,
               my_y,
               route_table_2d,
               Routing2DCodec::mcast_tree_y_offset(y_size, x_size),
               "Y",
               failure) &&
           embed_one_axis(
               mesh_graph,
               mesh_id,
               x_topo,
               my_x,
               route_table_2d,
               Routing2DCodec::mcast_tree_x_offset(y_size, x_size),
               "X",
               failure);
}

}  // namespace tt::tt_fabric
