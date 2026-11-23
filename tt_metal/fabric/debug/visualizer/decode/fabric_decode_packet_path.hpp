// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A ring buffer slot's path: the hops its packet's route takes, from the receiver that reads it next. The path decodes
// the route as written; it doesn't model the routers. Each hop follows its action, and each router's live mask,
// forwarding disabled and exit check are left for the visualizer to show beside the hop, through the node's router.
//
// A path is a list of nodes, each with its parent: a chain in 1D and for a 2D unicast, a tree for a 2D multicast.
//
//   { chip, router, vc, parent, via, reads, action, delivers, end }
//
//   chip      The chip the node is on. Null on a stop that has no chip.
//   router    The router whose receiver reads the hop, when it's on this rank.
//   vc        That receiver's VC, when decode knows it: the slot's VC at the first node, then the VC the parent's
//             receiver forwards on. Null once a hop has no router.
//   parent    The node this one came from. Null on the first node.
//   via       The direction the parent sent it in.
//   reads     2D: which map the hop's byte came from, "y" (at its chip's row) or "x" (at its column). Null in 1D.
//   action    What the hop reads: a 2D byte as names, or a 1D hop action. Null on a stop.
//   delivers  Whether the action writes to the chip.
//   end       Why the branch ends here (PathEnd), or null when it goes on.

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

#include <hostdevcommon/fabric_common.h>
#include <nlohmann/json.hpp>

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_packet_header.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_route_view.hpp"

namespace tt::tt_fabric::decode {

using json = nlohmann::ordered_json;

struct InterpretContext;

// Why a branch of a path ends. A stop is a node that reads nothing (its action is null): the visualizer still draws
// the arrow toward it.
enum class PathEnd : uint8_t {
    // The action delivers to the chip and sends the packet nowhere.
    DELIVERED,
    // The action is empty: a 2D byte with no direction or local bit, or a 1D noop.
    NO_ACTION,
    // A stop past the edge of a mesh that doesn't wrap that way. It has no chip.
    OFF_MESH,
    // A stop past a Z link decode can't follow: the chip's Z router isn't on this rank, or has no peer. It has no chip.
    UNKNOWN_CHIP,
    // A 2D stop at a chip and side the path already reached, so the route would repeat.
    LOOPS,
    // A stop past an intermesh link, whose far side rewrites the map as the packet lands. Its chip is the link's peer.
    LEAVES_MESH,
};

// Which map of a 2D route a hop's action byte comes from.
enum class RouteMap : uint8_t { Y, X };

// The ring buffer channel a slot is in.
struct SlotChannel {
    bool is_receiver = false;
    uint32_t vc = 0;
    uint32_t channel = 0;
};

// The channel of a ring buffer region's path, "channels/<senders|receivers>/vc<v>/ch<c>/ring_buffer" as
// for_each_router_region spells it. Null for any other path.
std::optional<SlotChannel> find_slot_channel(std::string_view path);

// The action byte a 2D router on `coord` facing `facing` reads, and the map it's from, as the router's codec picks it:
// an intermesh ingress receiver reads Y first (Routing2DCodec::decode_action_y_first); otherwise an E/W-facing router,
// or a Z-facing one on a mesh whose express axis is X, reads X, and any other reads Y first
// (Routing2DCodec::decode_action, with axis Y when the mesh has no express axis). A coordinate past a map reads 0.
struct Action2DRead {
    uint8_t action = 0;
    RouteMap map = RouteMap::Y;
};
Action2DRead read_2d_action(
    const Route2DMaps& maps,
    MeshCoord coord,
    eth_chan_directions facing,
    std::optional<Fabric2DAxis> express_axis,
    bool intermesh_ingress);

// The side a 1D forward leaves `chip` by, arriving with `facing`: straight through, out the side opposite. On a mesh
// that wraps around, a chip with exactly two neighbors has its two routers wired together, so a forward arriving from
// one leaves toward the other (FabricBuilder::get_router_connection_pairs).
eth_chan_directions find_1d_exit_direction(const RouteView& routes, FabricNodeId chip, eth_chan_directions facing);

// A 1D slot's path: a chain, node i reading actions[i], each forward leaving by find_1d_exit_direction. Null when
// context.path isn't a ring buffer or the slot's router isn't in context.routes.
json make_1d_path_json(
    std::span<const RoutingFieldsConstants::LowLatency::HopAction> actions, const InterpretContext& context);

// A 2D slot's path: breadth first, each direction in a hop's byte, lowest first, a child. Null as in 1D.
json make_2d_path_json(const Route2DMaps& maps, const InterpretContext& context);

}  // namespace tt::tt_fabric::decode
