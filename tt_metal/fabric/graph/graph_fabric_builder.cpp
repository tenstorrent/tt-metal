// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "graph_fabric_builder.hpp"

#include <algorithm>

#include <enchantum/enchantum.hpp>
#include <tt_stl/assert.hpp>

#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"

namespace tt::tt_fabric::graph {

GraphFabricBuilder::GraphFabricBuilder(
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Program& program,
    FabricContext& fabric_context,
    CoreType dispatch_core_type,
    const PhysicalGraph& graph) :
    FabricBuilder(device, program, fabric_context, dispatch_core_type), graph_(graph) {
    for (RouterIndex r = 0; r < graph_.logical.routers.size(); ++r) {
        if (graph_.logical.routers[r].port.node == local_node_) {
            on_chip_.emplace(graph_.eth_chans[r], r);
        }
    }
}

RouterVcShape GraphFabricBuilder::shape_of(RouterIndex router) const {
    RouterVcShape shape;
    const auto& counts = graph_.sender_counts[router];
    shape.num_vcs = static_cast<uint32_t>(counts.size());
    std::copy(counts.begin(), counts.end(), shape.sender_counts.begin());
    for (ChannelIndex c : graph_.logical.routers[router].channels) {
        const auto& ch = graph_.logical.channels[c];
        if (ch.kind == ChannelKind::Receiver) {
            shape.receiver_counts[ch.vc] = 1;
        }
    }
    finalize_vc_shape_bases(shape);
    return shape;
}

// layouts and per-slot sender flags; a slot no channel takes is not serviced
RouterGraphInputs GraphFabricBuilder::inputs_of(RouterIndex router) const {
    const auto& g = graph_.logical;
    const auto& counts = graph_.sender_counts[router];
    std::vector<uint32_t> base(counts.size(), 0);
    for (size_t vc = 1; vc < counts.size(); ++vc) {
        base[vc] = base[vc - 1] + counts[vc - 1];
    }
    const size_t num_senders = base.back() + counts.back();
    RouterGraphInputs in{
        .layout = graph_.layouts[router],
        .injection = std::vector<bool>(num_senders, false),
        .serviced = std::vector<bool>(num_senders, false)};
    for (ChannelIndex c : g.channels_of(router, ChannelKind::Sender)) {
        const uint32_t i = base[g.channels[c].vc] + graph_.index_in_vc[c];
        in.injection[i] = graph_.enters_ring[c];
        in.serviced[i] = true;
    }
    return in;
}

ComputeMeshRouterBuilder& GraphFabricBuilder::builder_of(RouterIndex router) {
    auto* builder = dynamic_cast<ComputeMeshRouterBuilder*>(routers_.at(graph_.eth_chans[router]).get());
    TT_FATAL(
        builder != nullptr, "fabric graph: router {} is not a compute mesh router", graph_.logical.routers[router].id);
    return *builder;
}

// the graph router bound to this eth channel; discovery must agree with the graph
std::unique_ptr<FabricRouterBuilder> GraphFabricBuilder::make_router(const RouterLocation& location) {
    const auto it = on_chip_.find(location.eth_chan);
    TT_FATAL(
        it != on_chip_.end(),
        "fabric graph: chip {} has a live eth channel {} facing {} but no graph router on it",
        local_node_,
        location.eth_chan,
        enchantum::to_string(location.direction));
    const RouterIndex r = it->second;
    const auto& router = graph_.logical.routers[r];
    TT_FATAL(
        *router.port.direction == location.direction,
        "fabric graph: router {} is on eth channel {}, which discovery has facing {}",
        router.id,
        location.eth_chan,
        enchantum::to_string(location.direction));
    TT_FATAL(router.peer.has_value(), "fabric graph: router {} has no link peer", router.id);
    TT_FATAL(
        router.port.peer == location.remote_node,
        "fabric graph: router {} links to {}, but the live link goes to {}",
        router.id,
        router.port.peer,
        location.remote_node);
    const auto& live_capability = chip_facts_.per_direction_capabilities.at(location.direction);
    TT_FATAL(
        live_capability == router.port.capability,
        "fabric graph: router {} is {}, but discovery classifies its link {}",
        router.id,
        enchantum::to_string(router.port.capability),
        live_capability ? enchantum::to_string(*live_capability) : "unclassified");
    return ComputeMeshRouterBuilder::build(
        fabric_context_,
        dispatch_core_type_,
        device_,
        program_,
        local_node_,
        location,
        chip_facts_,
        nullptr,
        RouterArchetype{.shape = shape_of(r), .turns = {}},
        inputs_of(r));
}

KernelCreationContext GraphFabricBuilder::kernel_creation_context() const {
    uint32_t mask = 0;
    for (const auto& [chan, r] : on_chip_) {
        mask |= 1u << chan;
    }
    return KernelCreationContext{
        .is_2D_routing = fabric_context_.is_2D_routing_enabled(),
        .master_router_chan = master_router_chan_,
        .num_local_fabric_routers = on_chip_.size(),
        .router_channels_mask = mask,
    };
}

// one connection per graph noc edge, into the slot the graph gave it; then the graph's link pairings
void GraphFabricBuilder::connect_routers() {
    TT_FATAL(
        routers_.size() == on_chip_.size(),
        "fabric graph: chip {} has {} graph routers but {} live eth channels",
        local_node_,
        on_chip_.size(),
        routers_.size());
    const auto& g = graph_.logical;
    for (const auto& e : g.edges) {
        const RouterIndex up = g.channels[e.from].router;
        const RouterIndex down = g.channels[e.to].router;
        if (e.transport == Transport::Noc && g.routers[up].port.node == local_node_) {
            builder_of(up).connect_to_downstream_channel(builder_of(down), g.channels[e.to].vc, graph_.index_in_vc[e.to]);
        }
    }
    for (const auto& link : graph_.links) {
        if (g.routers[link.a].port.node == local_node_) {
            builder_of(link.a).apply_link_settings(
                builder_of(link.b),
                link.noc_vc,
                link.link,
                link.num_links,
                fabric_context_.get_fabric_topology(),
                fabric_context_.is_ubb_galaxy());
        }
    }
}

}  // namespace tt::tt_fabric::graph
