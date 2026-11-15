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
        const auto& port = graph_.logical.routers[r].port;
        if (port.node == local_node_) {
            on_chip_.emplace(std::pair{*port.direction, port.plane}, r);
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

// the built router must have its channels where the graph put them
void GraphFabricBuilder::check_layout(RouterIndex router, const FabricRouterBuilder& built) const {
    const auto layout = built.get_channel_layout();
    TT_FATAL(layout.has_value(), "fabric graph: router {} has no channel layout", graph_.logical.routers[router].id);
    for (ChannelIndex c : graph_.logical.routers[router].channels) {
        const auto& ch = graph_.logical.channels[c];
        const auto& want = graph_.addresses[c];
        const auto& have = ch.kind == ChannelKind::Sender ? layout->senders[ch.vc] : layout->receivers[ch.vc];
        const uint32_t i = ch.kind == ChannelKind::Sender ? graph_.index_in_vc[c] : 0;
        TT_FATAL(
            i < have.size() && have[i].base_address == want.base && have[i].num_slots == want.slots,
            "fabric graph: {} vc{} {} is at {:#x} x{} in the graph but not in the built router",
            graph_.logical.routers[router].id,
            ch.vc,
            channel_name(ch),
            want.base,
            want.slots);
    }
}

ComputeMeshRouterBuilder& GraphFabricBuilder::builder_of(RouterIndex router) {
    return static_cast<ComputeMeshRouterBuilder&>(*routers_.at(eth_chan_.at(router)));
}

// the graph router on this live eth channel, with the graph's shape
std::unique_ptr<FabricRouterBuilder> GraphFabricBuilder::make_router(const RouterLocation& location) {
    const auto& chans = channels_by_direction_.at(location.direction);
    const auto plane = static_cast<uint32_t>(std::ranges::find(chans, location.eth_chan) - chans.begin());
    const auto it = on_chip_.find({location.direction, plane});
    TT_FATAL(
        it != on_chip_.end(),
        "fabric graph: chip {} has a live eth channel facing {} plane {} but no graph router",
        local_node_,
        enchantum::to_string(location.direction),
        plane);
    const RouterIndex r = it->second;
    const auto& router = graph_.logical.routers[r];
    TT_FATAL(
        router.port.peer == location.remote_node,
        "fabric graph: router {} links to {}, but the live link goes to {}",
        router.id,
        router.port.peer,
        location.remote_node);
    auto built = ComputeMeshRouterBuilder::build(
        fabric_context_,
        dispatch_core_type_,
        device_,
        program_,
        local_node_,
        location,
        chip_facts_,
        nullptr,
        RouterArchetype{.shape = shape_of(r), .turns = {}});
    check_layout(r, *built);
    eth_chan_[r] = location.eth_chan;
    return built;
}

// one connection per graph noc edge, into the slot the graph gave it; then the graph's link pairings
void GraphFabricBuilder::connect_routers() {
    TT_FATAL(
        eth_chan_.size() == on_chip_.size(),
        "fabric graph: chip {} has {} graph routers but {} live eth channels",
        local_node_,
        on_chip_.size(),
        eth_chan_.size());
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
