// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/fabric_manifest_collector.hpp"

#include <fmt/format.h>
#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>

#include <algorithm>
#include <array>

#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "tt_metal/fabric/builder/fabric_stream_assignment.hpp"
#include "tt_metal/fabric/builder/protected_domain_effect.hpp"
#include "tt_metal/fabric/builder/router_wiring_rules.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_router_builder.hpp"

namespace tt::tt_fabric {

namespace {

using NamedArgs = std::unordered_map<std::string, uint32_t>;

const FabricBuilderContext& builder_context() {
    return tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context().get_builder_context();
}

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

// A credit counter array's base address and the compile-time argument that passes it to the kernel.
struct CounterBase {
    const char* ct_arg_name;
    size_t address;
};

// Return the router's four L1 credit counter arrays. The kernel sends each pair to the peer in one packet,
// so it relies on the arrays being back to back and the same size; the size comes from that spacing.
manifest::L1CreditCounters collect_credit_counters(
    const FabricEriscDatamoverBuilder& erisc_builder, const std::vector<NamedArgs>& named_ct_args_per_risc) {
    const auto& config = erisc_builder.config;
    // In L1 order
    const std::array<CounterBase, 4> bases = {{
        {"TO_SENDER_REMOTE_ACK_COUNTERS_BASE_ADDR", config.to_sender_channel_remote_ack_counters_base_addr},
        {"TO_SENDER_REMOTE_COMPLETION_COUNTERS_BASE_ADDR",
         config.to_sender_channel_remote_completion_counters_base_addr},
        {"LOCAL_RECEIVER_ACK_COUNTERS_BASE_ADDR", config.receiver_channel_remote_ack_counters_base_addr},
        {"LOCAL_RECEIVER_COMPLETION_COUNTERS_BASE_ADDR", config.receiver_channel_remote_completion_counters_base_addr},
    }};

    // Check that the compile-time arguments match the config
    for (const auto& args : named_ct_args_per_risc) {
        for (const auto& base : bases) {
            TT_FATAL(
                get_named_arg(args, base.ct_arg_name) == base.address,
                "Fabric manifest: {} is {:#x}, but the router config has {:#x}",
                base.ct_arg_name,
                get_named_arg(args, base.ct_arg_name),
                base.address);
        }
    }

    // Check that the credit counter arrays have uniform size and are ordered u32 arrays
    const size_t size = bases[1].address - bases[0].address;
    for (size_t i = 1; i < bases.size(); ++i) {
        TT_FATAL(
            bases[i].address > bases[i - 1].address && bases[i].address - bases[i - 1].address == size &&
                size % sizeof(uint32_t) == 0,
            "Fabric manifest: credit counter array {} at {:#x} does not follow {} at {:#x} as a {}-byte u32 array",
            bases[i].ct_arg_name,
            bases[i].address,
            bases[i - 1].ct_arg_name,
            bases[i - 1].address,
            size);
    }

    const auto addresses_to_clear = builder_context().get_fabric_router_addresses_to_clear();
    const auto counter_array = [&](size_t address) {
        return manifest::L1Region{
            .address = static_cast<uint32_t>(address),
            .size = static_cast<uint32_t>(size),
            .num_elements = static_cast<uint32_t>(size / sizeof(uint32_t)),
            .size_per_element = static_cast<uint32_t>(sizeof(uint32_t)),
            .schema = "u32",
            .host_cleared = std::ranges::find(addresses_to_clear, address) != addresses_to_clear.end(),
        };
    };
    return {
        .to_sender_ack = counter_array(bases[0].address),
        .to_sender_completion = counter_array(bases[1].address),
        .receiver_ack = counter_array(bases[2].address),
        .receiver_completion = counter_array(bases[3].address),
    };
}

// Every RISC's VC*_USES_COUNTER_CREDITS arguments must match the mesh's credit plan, which the writer
// serializes as the mesh's credit_transport.
void check_credit_transport_args(
    const FabricEriscDatamoverBuilder& erisc_builder, const std::vector<NamedArgs>& named_ct_args_per_risc) {
    // Credit transport plan
    const auto& plan = builder_context().get_stream_assignment(erisc_builder.local_fabric_node_id.mesh_id).plan();

    // Check that the compile-time arguments match the credit transport plan
    for (const auto& args : named_ct_args_per_risc) {
        for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
            const auto name = fmt::format("VC{}_USES_COUNTER_CREDITS", vc);
            TT_FATAL(
                (get_named_arg(args, name) != 0) == plan.vc_uses_counters(vc),
                "Fabric manifest: {} is {}, but the mesh's credit plan has VC{} on {}",
                name,
                get_named_arg(args, name),
                vc,
                plan.vc_uses_counters(vc) ? "L1 counters" : "stream registers");
        }
    }
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
    check_credit_transport_args(erisc_builder, named_ct_args_per_risc);

    return {
        .identity = collect_identity(location),
        .link = collect_link(erisc_builder, location, chip_facts),
        .shape = collect_shape(erisc_builder, vc_shape, named_ct_args_per_risc),
        .credit_counters = collect_credit_counters(erisc_builder, named_ct_args_per_risc),
    };
}

}  // namespace tt::tt_fabric
