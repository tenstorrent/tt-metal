// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_graph_types.hpp"

#include <enchantum/enchantum.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_fabric::graph {

std::string router_id(const FabricNodeId& node, PortDirection direction, uint32_t plane) {
    return "m" + std::to_string(*node.mesh_id) + ".c" + std::to_string(node.chip_id) + "." +
           std::string(enchantum::to_string(*direction)) + ".p" + std::to_string(plane);
}

std::string channel_name(const LogicalChannel& channel) {
    if (channel.kind == ChannelKind::Receiver) {
        return "rx";
    }
    return channel.from ? std::string(enchantum::to_string(**channel.from)) : "worker";
}

std::optional<RouterIndex> LogicalGraph::find_router(std::string_view id) const {
    for (RouterIndex r = 0; r < routers.size(); ++r) {
        if (routers[r].id == id) {
            return r;
        }
    }
    return std::nullopt;
}

std::optional<ChannelIndex> LogicalGraph::receiver(RouterIndex router, uint32_t vc) const {
    for (ChannelIndex c : routers.at(router).channels) {
        if (channels[c].kind == ChannelKind::Receiver && channels[c].vc == vc) {
            return c;
        }
    }
    return std::nullopt;
}

std::vector<ChannelIndex> LogicalGraph::channels_of(RouterIndex router, ChannelKind kind) const {
    std::vector<ChannelIndex> out;
    for (ChannelIndex c : routers.at(router).channels) {
        if (channels[c].kind == kind) {
            out.push_back(c);
        }
    }
    return out;
}

std::vector<const LogicalEdge*> LogicalGraph::edges_from(ChannelIndex channel) const {
    std::vector<const LogicalEdge*> out;
    for (const auto& e : edges) {
        if (e.from == channel) {
            out.push_back(&e);
        }
    }
    return out;
}

std::vector<const LogicalEdge*> LogicalGraph::edges_into(ChannelIndex channel) const {
    std::vector<const LogicalEdge*> out;
    for (const auto& e : edges) {
        if (e.to == channel) {
            out.push_back(&e);
        }
    }
    return out;
}

}  // namespace tt::tt_fabric::graph
