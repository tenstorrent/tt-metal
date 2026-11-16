// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "graph_fabric_build_session.hpp"

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt_stl/assert.hpp>

#include "fabric_graph_inputs.hpp"
#include "fabric_wiring_rules.hpp"
#include "graph_fabric_builder.hpp"
#include "logical_graph_lowering.hpp"
#include "logical_graph_validation.hpp"
#include "physical_graph_lowering.hpp"
#include "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"

namespace tt::tt_fabric::graph {

namespace {

const FabricStaticSizedChannelsAllocator& stock_allocator(const FabricContext& fabric_context) {
    const auto& cfg = fabric_context.get_builder_context().get_fabric_router_config();
    const auto* allocator = dynamic_cast<const FabricStaticSizedChannelsAllocator*>(cfg.channel_allocator.get());
    TT_FATAL(allocator != nullptr, "fabric graph: stock router config has no static-sized channel allocator");
    return *allocator;
}

std::map<MeshId, ExpressAxis> express_axes(const ControlPlane& control_plane) {
    std::map<MeshId, ExpressAxis> axes;
    for (MeshId mesh : control_plane.get_mesh_graph().get_mesh_ids()) {
        if (const ExpressAxis axis = express_axis_of(control_plane, mesh); axis != ExpressAxis::NONE) {
            axes[mesh] = axis;
        }
    }
    return axes;
}

// the graph's link peers are the control plane's, down to the eth channel
void check_link_peers(const PhysicalGraph& graph, const ControlPlane& control_plane) {
    const auto& g = graph.logical;
    for (RouterIndex r = 0; r < g.routers.size(); ++r) {
        const auto peer = g.routers[r].peer;
        const auto live = control_plane.try_get_connected_mesh_chip_chan_ids(g.routers[r].port.node, graph.eth_chans[r]);
        if (!peer || !live) {
            continue;
        }
        TT_FATAL(
            live->first == g.routers[*peer].port.node && live->second == graph.eth_chans[*peer],
            "fabric graph: router {} on eth channel {} links to {} channel {}, the control plane says {} channel {}",
            g.routers[r].id,
            graph.eth_chans[r],
            g.routers[*peer].port.node,
            graph.eth_chans[*peer],
            live->first,
            live->second);
    }
}

}  // namespace

GraphFabricBuildSession::GraphFabricBuildSession(FabricContext& fabric_context) {
    if (fabric_context.is_tensix_enabled()) {
        log_info(tt::LogFabric, "fabric graph: tensix fabric, using the stock builder");
        return;
    }
    const Topology topology = fabric_context.get_fabric_topology();
    const auto& control_plane = fabric_context.get_control_plane();
    const auto express_axis = express_axes(control_plane);
    // corners fold only in 1D, which is single-mesh
    std::set<MeshId> wrap_around_meshes;
    if (!is_2D_topology(topology)) {
        for (MeshId mesh : control_plane.get_mesh_graph().get_mesh_ids()) {
            if (fabric_context.is_wrap_around_mesh(mesh)) {
                wrap_around_meshes.insert(mesh);
            }
        }
    }

    // logical: topology and turns
    const auto ports = with_live_planes(ports_from_mesh_graph(control_plane.get_mesh_graph(), express_axis), control_plane);
    const WiringOptions options{
        .topology = topology,
        .vc_config = fabric_context.get_builder_context().get_intermesh_vc_config(),
        .express_axis = express_axis,
        .wrap_around_meshes = wrap_around_meshes};
    const auto rules = stock_wiring_rules(ports, options);
    auto logical = lower_to_logical(ports, *rules);
    validate(logical, *rules, ValidationLimits{});

    // channel graph: each router's slots, and which senders enter a protected ring
    auto channels = lower_to_channels(std::move(logical), is_2D_topology(topology));
    legacy_stock_sender_counts(channels, options);  // stock-sized routers until #65720
    mark_ring_entries(channels, options, [&control_plane](const FabricNodeId& node) {
        return make_protected_ring_queries(control_plane, node);
    });

    // physical: addresses from the stock allocator, then the legacy link pairings
    const auto& reference = stock_allocator(fabric_context);
    graph_ = lower_to_physical(std::move(channels), [&reference](const auto& sender_counts) {
        return FabricStaticSizedChannelsAllocator(reference, sender_counts).get_layout();
    });
    graph_->eth_chans = bind_eth_channels(graph_->logical, [&control_plane](const FabricNodeId& node, RoutingDirection d) {
        return control_plane.get_active_fabric_eth_routing_planes_in_direction(node, d);
    });
    check_link_peers(*graph_, control_plane);
    graph_->links = legacy_link_pairings(graph_->logical, topology, wrap_around_meshes);
    graph_->handshake_masters = legacy_handshake_masters(graph_->logical, graph_->eth_chans);

    log_info(
        tt::LogFabric,
        "fabric graph: {} routers, {} channels, {} edges, {} rings, {} VCs, {} link pairings",
        graph_->logical.routers.size(),
        graph_->logical.channels.size(),
        graph_->logical.edges.size(),
        graph_->logical.rings.size(),
        graph_->logical.vcs.size(),
        graph_->links.size());
}

std::unique_ptr<FabricBuilder> GraphFabricBuildSession::make_device_builder(
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Program& program,
    FabricContext& fabric_context,
    CoreType dispatch_core_type) const {
    if (!graph_) {
        return std::make_unique<FabricBuilder>(device, program, fabric_context, dispatch_core_type);
    }
    return std::make_unique<GraphFabricBuilder>(device, program, fabric_context, dispatch_core_type, *graph_);
}


}  // namespace tt::tt_fabric::graph
