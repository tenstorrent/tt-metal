// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_packet_path.hpp"

#include <bit>
#include <charconv>
#include <set>
#include <system_error>
#include <utility>
#include <vector>

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_interpret.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_routing_table.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_paths.hpp"

namespace tt::tt_fabric::decode {

namespace {

using HopAction = RoutingFieldsConstants::LowLatency::HopAction;

// One node of a path: a hop, which reads the route on its chip, or a stop (end set before it's read).
struct PathNode {
    std::optional<FabricNodeId> chip;
    // The side the packet arrives on: the facing of the router that reads the hop.
    eth_chan_directions facing = eth_chan_directions::EAST;
    const RouteRouter* router = nullptr;
    std::optional<uint32_t> vc;
    std::optional<uint32_t> parent;
    std::optional<eth_chan_directions> via;
    std::optional<RouteMap> reads;
    // Null until the hop is read, and on a stop.
    json action = nullptr;
    bool delivers = false;
    std::optional<PathEnd> end;
};

PathNode make_stop(std::optional<FabricNodeId> chip, PathEnd end) { return PathNode{.chip = chip, .end = end}; }

// The node a packet sent `direction` from `chip` reaches, received on VC `vc`: a hop onto the chip one coordinate step
// that way, arriving with the opposite facing, read by that chip's router on `routing_plane` when it's on this rank.
// For Z, the far end of the chip's Z link. A stop when there's no such chip, or the link leaves the mesh.
PathNode follow_link(
    const RouteView& routes,
    FabricNodeId chip,
    eth_chan_directions direction,
    routing_plane_id_t routing_plane,
    std::optional<uint32_t> vc) {
    if (direction == eth_chan_directions::Z) {
        const RouteRouter* z_router = routes.find_receiving_router(chip, routing_plane, eth_chan_directions::Z);
        if (z_router == nullptr || !z_router->peer.has_value()) {
            return make_stop(std::nullopt, PathEnd::UNKNOWN_CHIP);
        }
        if (z_router->is_intermesh) {
            return make_stop(z_router->peer->node, PathEnd::LEAVES_MESH);
        }
        return PathNode{
            .chip = z_router->peer->node,
            .facing = eth_chan_directions::Z,
            .router = routes.find_router(*z_router->peer),
            .vc = vc};
    }
    const std::optional<FabricNodeId> next_chip = routes.find_next_chip(chip, direction);
    if (!next_chip.has_value()) {
        return make_stop(std::nullopt, PathEnd::OFF_MESH);
    }
    const eth_chan_directions facing = get_opposite_direction(direction);
    return PathNode{
        .chip = next_chip,
        .facing = facing,
        .router = routes.find_receiving_router(*next_chip, routing_plane, facing),
        .vc = vc};
}

// The first node, and what reading it needs beyond the node: its receiver channel, and whether that's an intermesh
// ingress receiver.
struct FirstNode {
    PathNode node;
    uint32_t receiver_channel = 0;
    bool intermesh_ingress = false;
};

// Who reads the slot's route next. A receiver slot: its own router, on the slot's VC. A sender slot: the receiver at
// the far end of its router's link, on the same VC; a stop when the link leaves the mesh (its far side rewrites the
// map) or goes nowhere decode can place.
std::optional<FirstNode> find_first_node(const InterpretContext& context) {
    const std::optional<SlotChannel> slot = find_slot_channel(context.path);
    const RouteRouter* router = context.routes.find_router(context.router_ref);
    if (!slot.has_value() || router == nullptr) {
        return std::nullopt;
    }
    const FabricNodeId chip = context.router_ref.node;
    if (slot->is_receiver) {
        const bool has_receiver =
            slot->vc < router->receivers.size() && slot->channel < router->receivers[slot->vc].size();
        return FirstNode{
            .node = PathNode{.chip = chip, .facing = router->facing, .router = router, .vc = slot->vc},
            .receiver_channel = slot->channel,
            .intermesh_ingress = has_receiver && router->receivers[slot->vc][slot->channel].intermesh_ingress};
    }
    if (router->is_intermesh) {
        const std::optional<FabricNodeId> peer_chip =
            router->peer.has_value() ? std::optional(router->peer->node) : std::nullopt;
        return FirstNode{.node = make_stop(peer_chip, PathEnd::LEAVES_MESH)};
    }
    return FirstNode{
        .node = follow_link(context.routes, chip, router->facing, context.router_ref.key.routing_plane, slot->vc)};
}

// The VC a hop's receiver forwards on, which its children are received on. Each VC has one receiver channel, so a
// hop's receiver is channel 0 of its VC; the first node's is its slot's. Null when the hop has no router or no VC.
std::optional<uint32_t> find_forwarding_vc(const PathNode& node, uint32_t receiver_channel) {
    if (node.router == nullptr || !node.vc.has_value()) {
        return std::nullopt;
    }
    const std::vector<std::vector<RouteReceiver>>& receivers = node.router->receivers;
    if (*node.vc >= receivers.size() || receiver_channel >= receivers[*node.vc].size()) {
        return std::nullopt;
    }
    return receivers[*node.vc][receiver_channel].forwards_on;
}

// Reads the next number of `text`, and moves past it. Null when `text` doesn't start with one.
std::optional<uint32_t> consume_number(std::string_view& text) {
    uint32_t number = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), number);
    if (error != std::errc{} || end == text.data()) {
        return std::nullopt;
    }
    text.remove_prefix(static_cast<size_t>(end - text.data()));
    return number;
}

// Moves past `prefix` when `text` starts with it.
bool consume_prefix(std::string_view& text, std::string_view prefix) {
    if (!text.starts_with(prefix)) {
        return false;
    }
    text.remove_prefix(prefix.size());
    return true;
}

json make_node_json(const PathNode& node) {
    json out;
    out["chip"] = node.chip.has_value() ? json(manifest::chip_path(*node.chip)) : json(nullptr);
    out["router"] = node.router != nullptr ? json(manifest::router_path(node.router->ref)) : json(nullptr);
    out["vc"] = node.vc.has_value() ? json(*node.vc) : json(nullptr);
    out["parent"] = node.parent.has_value() ? json(*node.parent) : json(nullptr);
    out["via"] = node.via.has_value() ? json(manifest::lower_enum_name(*node.via)) : json(nullptr);
    out["reads"] = node.reads.has_value() ? json(manifest::lower_enum_name(*node.reads)) : json(nullptr);
    out["action"] = node.action;
    out["delivers"] = node.delivers;
    out["end"] = node.end.has_value() ? json(manifest::lower_enum_name(*node.end)) : json(nullptr);
    return out;
}

json make_nodes_json(const std::vector<PathNode>& nodes) {
    json out = json::array();
    for (const PathNode& node : nodes) {
        out.push_back(make_node_json(node));
    }
    return out;
}

}  // namespace

std::optional<SlotChannel> find_slot_channel(std::string_view path) {
    SlotChannel slot;
    if (!consume_prefix(path, "channels/")) {
        return std::nullopt;
    }
    if (consume_prefix(path, "receivers/vc")) {
        slot.is_receiver = true;
    } else if (!consume_prefix(path, "senders/vc")) {
        return std::nullopt;
    }
    const std::optional<uint32_t> vc = consume_number(path);
    if (!vc.has_value() || !consume_prefix(path, "/ch")) {
        return std::nullopt;
    }
    const std::optional<uint32_t> channel = consume_number(path);
    if (!channel.has_value() || path != "/ring_buffer") {
        return std::nullopt;
    }
    slot.vc = *vc;
    slot.channel = *channel;
    return slot;
}

Action2DRead read_2d_action(
    const Route2DMaps& maps,
    MeshCoord coord,
    eth_chan_directions facing,
    std::optional<Fabric2DAxis> express_axis,
    bool intermesh_ingress) {
    const uint8_t y_action = coord.y < maps.y.size() ? maps.y[coord.y] : 0;
    const uint8_t x_action = coord.x < maps.x.size() ? maps.x[coord.x] : 0;
    const bool reads_x_only =
        !intermesh_ingress && (facing == eth_chan_directions::EAST || facing == eth_chan_directions::WEST ||
                               (facing == eth_chan_directions::Z && express_axis == Fabric2DAxis::X));
    if (reads_x_only || y_action == 0) {
        return Action2DRead{.action = x_action, .map = RouteMap::X};
    }
    return Action2DRead{.action = y_action, .map = RouteMap::Y};
}

eth_chan_directions find_1d_exit_direction(const RouteView& routes, FabricNodeId chip, eth_chan_directions facing) {
    const RouteMesh* mesh = routes.find_mesh(chip.mesh_id);
    const uint8_t neighbors = routes.find_neighbor_directions(chip);
    const uint8_t facing_bit = facing < eth_chan_directions::COUNT ? Routing2DCodec::action_bit(facing) : 0;
    if (mesh != nullptr && mesh->wraps_around && std::popcount(neighbors) == 2 && (neighbors & facing_bit) != 0) {
        return static_cast<eth_chan_directions>(std::countr_zero(static_cast<uint8_t>(neighbors & ~facing_bit)));
    }
    return get_opposite_direction(facing);
}

json make_1d_path_json(std::span<const HopAction> actions, const InterpretContext& context) {
    const std::optional<FirstNode> first = find_first_node(context);
    if (!first.has_value()) {
        return json(nullptr);
    }
    const routing_plane_id_t routing_plane = context.router_ref.key.routing_plane;
    std::vector<PathNode> nodes{first->node};
    // A chain: each node reads the next action, and a forward adds the next node. The actions end with one that
    // doesn't forward, so the chain ends there at the latest.
    for (size_t index = 0; index < nodes.size() && index < actions.size() && !nodes[index].end.has_value(); ++index) {
        PathNode& node = nodes[index];
        const HopAction action = actions[index];
        node.action = detail::make_1d_hop_action_json(action);
        node.delivers = (action & RoutingFieldsConstants::LowLatency::WRITE_ONLY) != 0;
        if ((action & RoutingFieldsConstants::LowLatency::FORWARD_ONLY) == 0) {
            node.end = node.delivers ? PathEnd::DELIVERED : PathEnd::NO_ACTION;
            break;
        }
        const eth_chan_directions exit = find_1d_exit_direction(context.routes, *node.chip, node.facing);
        PathNode next = follow_link(
            context.routes,
            *node.chip,
            exit,
            routing_plane,
            find_forwarding_vc(node, index == 0 ? first->receiver_channel : 0));
        next.parent = static_cast<uint32_t>(index);
        next.via = exit;
        nodes.push_back(std::move(next));
    }
    return make_nodes_json(nodes);
}

json make_2d_path_json(const Route2DMaps& maps, const InterpretContext& context) {
    const std::optional<FirstNode> first = find_first_node(context);
    if (!first.has_value()) {
        return json(nullptr);
    }
    const RouteView& routes = context.routes;
    const routing_plane_id_t routing_plane = context.router_ref.key.routing_plane;
    std::vector<PathNode> nodes{first->node};
    // Each chip and side a hop reached. Finite, so the walk ends: a hop reaching one again is a loops stop.
    std::set<std::pair<FabricNodeId, eth_chan_directions>> reached;
    if (!first->node.end.has_value()) {
        reached.emplace(*first->node.chip, first->node.facing);
    }
    // Breadth first: nodes are read in the order they're added, so the list is the walk's queue.
    for (size_t index = 0; index < nodes.size(); ++index) {
        if (nodes[index].end.has_value()) {
            continue;
        }
        const FabricNodeId chip = *nodes[index].chip;
        const RouteMesh* mesh = routes.find_mesh(chip.mesh_id);
        const std::optional<MeshCoord> coord = routes.find_chip_coord(chip);
        const bool intermesh_ingress = index == 0 && first->intermesh_ingress;
        const Action2DRead read =
            coord.has_value() && mesh != nullptr
                ? read_2d_action(maps, *coord, nodes[index].facing, mesh->express_axis, intermesh_ingress)
                : Action2DRead{};
        const uint8_t directions = read.action & Routing2DCodec::ACTION_ETH_MASK;
        nodes[index].reads = read.map;
        nodes[index].action = make_2d_action_json(read.action);
        nodes[index].delivers = (read.action & Routing2DCodec::ACTION_LOCAL_DELIVER) != 0;
        if (directions == 0) {
            nodes[index].end = nodes[index].delivers ? PathEnd::DELIVERED : PathEnd::NO_ACTION;
            continue;
        }
        const std::optional<uint32_t> forwarding_vc =
            find_forwarding_vc(nodes[index], index == 0 ? first->receiver_channel : 0);
        for (uint32_t bit = 0; bit < eth_chan_directions::COUNT; ++bit) {
            if ((directions & (1u << bit)) == 0) {
                continue;
            }
            const auto direction = static_cast<eth_chan_directions>(bit);
            PathNode child = follow_link(routes, chip, direction, routing_plane, forwarding_vc);
            if (!child.end.has_value() && !reached.emplace(*child.chip, child.facing).second) {
                child = make_stop(child.chip, PathEnd::LOOPS);
            }
            child.parent = static_cast<uint32_t>(index);
            child.via = direction;
            nodes.push_back(std::move(child));
        }
    }
    return make_nodes_json(nodes);
}

}  // namespace tt::tt_fabric::decode
