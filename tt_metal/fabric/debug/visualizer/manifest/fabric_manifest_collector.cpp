// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_collector.hpp"

#include <fmt/format.h>
#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>

#include <numeric>

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
    TT_FATAL(it != args.end(), "Fabric manifest: missing fabric router named compile-time argument {}", name);
    return it->second;
}

// ERISC `risc_id`'s `name` argument must equal `expected`, the builder value the manifest expects.
void check_risc_named_arg(
    const std::vector<NamedArgs>& named_ct_args_per_risc, size_t risc_id, const std::string& name, uint32_t expected) {
    const uint32_t actual = get_named_arg(named_ct_args_per_risc.at(risc_id), name);
    TT_FATAL(
        actual == expected,
        "Fabric manifest: ERISC{} {} is {}, but the builder has {}",
        risc_id,
        name,
        actual,
        expected);
}

// Every RISC's `name` argument must equal `expected`, the builder value the manifest records.
void check_named_arg(const std::vector<NamedArgs>& named_ct_args_per_risc, const std::string& name, uint32_t expected) {
    for (size_t risc_id = 0; risc_id < named_ct_args_per_risc.size(); ++risc_id) {
        check_risc_named_arg(named_ct_args_per_risc, risc_id, name, expected);
    }
}

// A value the builder decides only while emitting compile-time arguments, so it is read back from them. Every
// RISC must receive the same value.
uint32_t emitted_value(const std::vector<NamedArgs>& named_ct_args_per_risc, const std::string& name) {
    const uint32_t value = get_named_arg(named_ct_args_per_risc.front(), name);
    check_named_arg(named_ct_args_per_risc, name, value);
    return value;
}

bool emitted_flag(const std::vector<NamedArgs>& named_ct_args_per_risc, const std::string& name) {
    return emitted_value(named_ct_args_per_risc, name) != 0;
}

// Return the RouterIdentity based on its location.
manifest::RouterIdentity collect_identity(const RouterLocation& location) {
    return {
        .eth_chan = location.eth_chan,
    };
}

// Return the EthLink based on the router's location and chip-wide facts. The peer the router was built toward
// must be the one ControlPlane connects its channel to.
manifest::EthLink collect_link(const ManifestRouterInputs& inputs) {
    const auto& location = inputs.location;
    const auto& capability = inputs.chip_facts.per_direction_capabilities.at(location.direction);
    TT_FATAL(
        capability.has_value(),
        "Fabric manifest: router on channel {} has no classified edge in the direction it faces",
        location.eth_chan);
    const auto peer = inputs.control_plane.try_get_connected_mesh_chip_chan_ids(
        inputs.erisc_builder.local_fabric_node_id, location.eth_chan);
    TT_FATAL(
        !peer.has_value() || peer->first == location.remote_node,
        "Fabric manifest: router on channel {} was built toward {}, but ControlPlane connects it to {}",
        location.eth_chan,
        location.remote_node,
        peer->first);
    return {
        .direction = builder::routing_direction_to_eth_direction(location.direction),
        .edge_capability = *capability,
        .is_dispatch_link = location.is_dispatch_link,
    };
}

// Return the RouterShape based on the erisc builder, its VC shape, and named compile-time arguments.
manifest::RouterShape collect_shape(const ManifestRouterInputs& inputs) {
    const auto& vc_shape = inputs.vc_shape;
    const auto& named_ct_args_per_risc = inputs.named_ct_args_per_risc;
    manifest::RouterShape shape{
        .num_vcs = vc_shape.num_vcs,
        .senders_per_vc = vc_shape.sender_counts,
        .receivers_per_vc = vc_shape.receiver_counts,
        .num_active_eriscs = static_cast<uint32_t>(inputs.erisc_builder.get_configured_risc_count()),
        .channel_trimming_overrides_applied = inputs.erisc_builder.get_channel_trimming_overrides().has_value(),
        .vc0_bubble_flow_control = emitted_flag(named_ct_args_per_risc, "ENABLE_DEADLOCK_AVOIDANCE"),
    };

    for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        check_named_arg(
            named_ct_args_per_risc, fmt::format("ACTUAL_VC{}_SENDER_CHANNELS", vc), shape.senders_per_vc[vc]);
    }
    check_named_arg(
        named_ct_args_per_risc,
        "NUM_RECEIVER_CHANNELS",
        std::accumulate(shape.receivers_per_vc.begin(), shape.receivers_per_vc.end(), 0u));
    return shape;
}

}  // namespace

manifest::Router collect_manifest_router(const ManifestRouterInputs& inputs) {
    TT_FATAL(
        inputs.named_ct_args_per_risc.size() == inputs.erisc_builder.get_configured_risc_count(),
        "Fabric manifest: got compile-time arguments for {} RISCs, but the router runs {}",
        inputs.named_ct_args_per_risc.size(),
        inputs.erisc_builder.get_configured_risc_count());

    return {
        .identity = collect_identity(inputs.location),
        .link = collect_link(inputs),
        .shape = collect_shape(inputs),
    };
}

}  // namespace tt::tt_fabric
