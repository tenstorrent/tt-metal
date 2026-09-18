// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/equal_cost_unicast.hpp>

#include <set>
#include <utility>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "fabric_context.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal::experimental::fabric {

std::optional<std::array<UnicastRoute, 2>> get_equal_cost_unicast_routes(
    const tt::tt_fabric::FabricNodeId& source,
    const tt::tt_fabric::FabricNodeId& destination,
    const std::array<tt::tt_fabric::FabricNodeId, 2>& neighbors) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& context = control_plane.get_fabric_context();
    if (!context.is_2D_routing_enabled() || source.mesh_id != destination.mesh_id || source == destination ||
        neighbors[0] == neighbors[1]) {
        return std::nullopt;
    }
    // Resolve paths using the same live routing tables as canonical unicast. The
    // application cannot request detours or supply packet-header instructions.
    auto shortest_path = [&](const tt::tt_fabric::FabricNodeId& start) {
        std::vector<tt::tt_fabric::FabricNodeId> path;
        if (start == destination) {
            return std::vector<tt::tt_fabric::FabricNodeId>{start};
        }
        for (const auto channel : control_plane.get_forwarding_eth_chans_to_chip(start, destination)) {
            const auto candidate = control_plane.get_fabric_route(start, destination, channel);
            std::vector<tt::tt_fabric::FabricNodeId> nodes{start};
            for (const auto& [node, _] : candidate) {
                if (node != nodes.back()) {
                    nodes.push_back(node);
                }
            }
            if (nodes.back() == destination && (path.empty() || nodes.size() < path.size())) {
                path = std::move(nodes);
            }
        }
        return path;
    };
    const auto canonical_path = shortest_path(source);
    if (canonical_path.empty()) {
        return std::nullopt;
    }
    auto route_via = [&](const tt::tt_fabric::FabricNodeId& neighbor) -> std::optional<UnicastRoute> {
        if (source.mesh_id != neighbor.mesh_id || source == neighbor ||
            tt::tt_fabric::get_neighbor_eth_directions(source, neighbor).empty()) {
            return std::nullopt;
        }

        auto path = shortest_path(neighbor);
        if (path.empty()) {
            return std::nullopt;
        }
        path.insert(path.begin(), source);
        if (path.size() != canonical_path.size()) {
            return std::nullopt;
        }
        if (path.size() < 2 || path.size() - 1 > context.get_2d_pkt_hdr_route_buffer_size()) {
            return std::nullopt;
        }

        std::set<tt::tt_fabric::FabricNodeId> visited{source};
        std::vector<tt::tt_fabric::eth_chan_directions> directions;
        for (size_t hop = 1; hop < path.size(); ++hop) {
            const auto& from = path[hop - 1];
            const auto& to = path[hop];
            if (to.mesh_id != source.mesh_id || !visited.insert(to).second) {
                return std::nullopt;
            }
            const auto direction = control_plane.get_forwarding_direction(from, to);
            if (!direction) {
                return std::nullopt;
            }
            const auto eth_direction = control_plane.routing_direction_to_eth_direction(*direction);
            if (eth_direction >= tt::tt_fabric::eth_chan_directions::Z) {
                return std::nullopt;
            }
            const auto channels = control_plane.get_active_fabric_eth_routing_planes_in_direction(from, *direction);
            if (channels.empty()) {
                return std::nullopt;
            }
            // Every plane on this egress must reach the same physical neighbor. Express links
            // require a route representation that also selects an edge, rather than a direction.
            for (const auto channel : channels) {
                const auto peer = control_plane.try_get_connected_mesh_chip_chan_ids(from, channel);
                if (!peer || peer->first != to) {
                    return std::nullopt;
                }
            }
            // The header has one branch offset per EW direction. Keep canonical routing for
            // paths requiring an NS-to-EW transition until this API can represent those turns.
            if (!directions.empty() &&
                (directions.back() == tt::tt_fabric::NORTH || directions.back() == tt::tt_fabric::SOUTH) &&
                (eth_direction == tt::tt_fabric::EAST || eth_direction == tt::tt_fabric::WEST)) {
                return std::nullopt;
            }
            directions.push_back(eth_direction);
        }

        constexpr uint32_t commands_per_word = 8;
        UnicastRoute route;
        route.args_ = {directions.front(), static_cast<uint32_t>(directions.size())};
        route.args_.resize(2 + (directions.size() + commands_per_word - 1) / commands_per_word, 0);
        for (size_t hop = 0; hop < directions.size(); ++hop) {
            // Injection performs the first hop. The final command drains through the ingress.
            const bool terminal = hop + 1 == directions.size();
            const auto direction = terminal ? directions.back() : directions[hop + 1];
            const bool ns = direction == tt::tt_fabric::NORTH || direction == tt::tt_fabric::SOUTH;
            // Reuse Fabric's canonical encoder for a one-hop segment with its forward
            // command prepended: [forward, drain]. Selecting the appropriate command
            // also supports folded physical paths without duplicating opcode mappings.
            uint8_t segment[2];
            tt::tt_fabric::routing_encoding::encode_2d_unicast(
                ns ? 1 : 0,
                ns ? 0 : 1,
                direction == tt::tt_fabric::SOUTH,
                direction == tt::tt_fabric::EAST,
                segment,
                2,
                true);
            route.args_[2 + hop / commands_per_word] |= static_cast<uint32_t>(segment[terminal ? 1 : 0])
                                                        << (4 * (hop % commands_per_word));
        }
        return route;
    };
    const auto first = route_via(neighbors[0]);
    const auto second = route_via(neighbors[1]);
    if (!first || !second || first->initial_direction() == second->initial_direction()) {
        return std::nullopt;
    }
    return std::array<UnicastRoute, 2>{*first, *second};
}

}  // namespace tt::tt_metal::experimental::fabric
