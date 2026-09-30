// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/fabric_manifest_collector.hpp"

#include <fmt/format.h>
#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>

#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "tt_metal/fabric/builder/protected_domain_effect.hpp"
#include "tt_metal/fabric/builder/router_wiring_rules.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/fabric_router_builder.hpp"

namespace tt::tt_fabric {

namespace {

using NamedArgs = std::unordered_map<std::string, uint32_t>;

// Helper to get a named compile-time argument.
uint32_t get_named_arg(const NamedArgs& args, const std::string& name) {
    const auto it = args.find(name);
    TT_FATAL(it != args.end(), "Missing fabric router named compile-time argument {}", name);
    return it->second;
}

// Return the RouterIdentity based on its location.
manifest::RouterIdentity collect_identity(const RouterLocation& location) {
    return {
        .eth_chan = location.eth_chan,
    };
}

// Return the EthLink based on the erisc builder, its location, and chip-wide facts.
manifest::EthLink collect_link(
    const FabricEriscDatamoverBuilder& erisc_builder,
    const RouterLocation& location,
    const ChipRoutingFacts& chip_facts) {
    const EdgeCapability edge_capability = chip_facts.per_direction_capabilities.at(location.direction).value();
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto peer =
        control_plane.try_get_connected_mesh_chip_chan_ids(erisc_builder.local_fabric_node_id, location.eth_chan);
    TT_FATAL(
        !peer.has_value() || peer->first == location.remote_node,
        "Fabric manifest: router on channel {} was built toward {}, but ControlPlane connects it to {}",
        location.eth_chan,
        location.remote_node,
        peer->first);
    return {
        .direction = builder::routing_direction_to_eth_direction(location.direction),
        .edge_capability = edge_capability,
        .dispatch_link = location.is_dispatch_link,
    };
}

// Return the RouterShape based on the erisc builder, its VC shape, and named compile-time arguments.
manifest::RouterShape collect_shape(
    const FabricEriscDatamoverBuilder& erisc_builder,
    const RouterVcShape& vc_shape,
    const std::vector<NamedArgs>& named_ct_args_per_risc) {
    manifest::RouterShape shape{
        .num_vcs = vc_shape.num_vcs,
        .senders_per_vc = vc_shape.sender_counts,
        .receivers_per_vc = vc_shape.receiver_counts,
        .num_active_eriscs = static_cast<uint32_t>(erisc_builder.get_configured_risc_count()),
        .channel_trimming_overrides_applied = erisc_builder.has_channel_trimming_overrides(),
        .vc0_bubble_flow_control = get_named_arg(named_ct_args_per_risc.front(), "ENABLE_DEADLOCK_AVOIDANCE") != 0,
    };

    for (const auto& args : named_ct_args_per_risc) {
        for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
            const auto name = fmt::format("ACTUAL_VC{}_SENDER_CHANNELS", vc);
            TT_FATAL(
                get_named_arg(args, name) == shape.senders_per_vc[vc],
                "Fabric manifest: {} is {}, but the router's shape has {} senders on VC{}",
                name,
                get_named_arg(args, name),
                shape.senders_per_vc[vc],
                vc);
        }
    }
    return shape;
}

}  // namespace

// Build a manifest Router using information from fabric builder.
manifest::Router collect_manifest_router(
    const FabricEriscDatamoverBuilder& erisc_builder,
    const RouterVcShape& vc_shape,
    const std::vector<NamedArgs>& named_ct_args_per_risc,
    const RouterLocation& location,
    const ChipRoutingFacts& chip_facts) {
    TT_FATAL(
        named_ct_args_per_risc.size() == erisc_builder.get_configured_risc_count(),
        "Fabric manifest: got compile-time arguments for {} RISCs, but the router runs {}",
        named_ct_args_per_risc.size(),
        erisc_builder.get_configured_risc_count());

    return {
        .identity = collect_identity(location),
        .link = collect_link(erisc_builder, location, chip_facts),
        .shape = collect_shape(erisc_builder, vc_shape, named_ct_args_per_risc),
    };
}

}  // namespace tt::tt_fabric
