// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/fabric_manifest_collector.hpp"

#include <fmt/format.h>
#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>

#include <algorithm>
#include <array>
#include <optional>
#include <variant>

#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp"
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

// Every RISC's `name` argument must equal `expected`, the builder value the manifest records.
void check_named_arg(const std::vector<NamedArgs>& named_ct_args_per_risc, const std::string& name, uint32_t expected) {
    for (size_t risc_id = 0; risc_id < named_ct_args_per_risc.size(); ++risc_id) {
        const uint32_t actual = get_named_arg(named_ct_args_per_risc[risc_id], name);
        TT_FATAL(
            actual == expected,
            "Fabric manifest: ERISC{} {} is {}, but the builder has {}",
            risc_id,
            name,
            actual,
            expected);
    }
}

manifest::L1Region l1_region(
    size_t address, size_t size, std::string schema, const std::vector<size_t>& addresses_to_clear) {
    return {
        .address = static_cast<uint32_t>(address),
        .size = static_cast<uint32_t>(size),
        .schema = std::move(schema),
        .host_cleared = std::ranges::find(addresses_to_clear, address) != addresses_to_clear.end(),
    };
}

// What the kernel reads for one sender channel. The kernel indexes its per-channel tables two ways (see
// RouterVcShape and StreamAssignment): the compact index, over this router's own counts, and the fabric
// position, over the fabric's maximum counts, which only the free-slots stream table uses.
struct SenderChannelIndex {
    uint32_t vc;
    uint32_t channel;
    uint32_t compact;
    uint32_t fabric_position;
};

// The builder facts shared by every sender channel on a router.
struct SenderCollectionContext {
    const FabricEriscDatamoverBuilder& erisc_builder;
    const FabricStaticSizedChannelsAllocator& allocator;
    const builder::RouterProducerSlots& producer_slots;
    const std::vector<NamedArgs>& named_ct_args_per_risc;
    // The stream register ids the builder assigned, by compile-time argument name.
    const NamedArgs& assigned_streams;
    const CreditTransportPlan& plan;
    const std::vector<size_t>& addresses_to_clear;
    bool vc0_bubble_flow_control;
};

// The stream register the builder assigned under `name`. Every RISC must receive the same id.
manifest::StreamRef assigned_stream(const SenderCollectionContext& ctx, const std::string& name) {
    const uint32_t stream_id = get_named_arg(ctx.assigned_streams, name);
    check_named_arg(ctx.named_ct_args_per_risc, name, stream_id);
    return {.stream_id = stream_id};
}

// The worker slot is fed by the local worker. Any other channel is fed by the sibling router in its producer
// slot, but only once that router has connected to it, which is what marks the channel's connection static.
std::optional<manifest::SenderChannelProducer> collect_sender_producer(
    const SenderCollectionContext& ctx, const SenderChannelIndex& index) {
    const bool static_connection =
        ctx.erisc_builder.sender_channel_connection_liveness_check_disable_array[index.compact];
    check_named_arg(
        ctx.named_ct_args_per_risc,
        fmt::format("SENDER_CH_{}_WAIT_STATIC_CONNECTION", index.compact),
        static_connection ? 1 : 0);

    // Make sure that the worker channel is not statically connected
    if (ctx.producer_slots.worker_channel(index.vc) == index.channel) {
        TT_FATAL(
            !static_connection,
            "Fabric manifest: VC{} channel {} is the worker channel, but a router connected to it",
            index.vc,
            index.channel);
        return manifest::LocalWorker{};
    }

    // We shouldn't get here because if the channel is not statically connected, it should be be a worker channel
    if (!static_connection) {
        return std::nullopt;
    }

    // Get the producer for the channel
    const auto producer = ctx.producer_slots.producer_at(index.vc, index.channel);
    TT_FATAL(
        producer.has_value(),
        "Fabric manifest: a router connected to VC{} channel {}, which is not a producer slot",
        index.vc,
        index.channel);
    return manifest::SiblingRouterRef{.direction = *producer};
}

// Stream-backed credits come from the TO_SENDER_* tables, which the kernel reads by compact index. Counter-backed
// credits are this channel's element of the to_sender counter arrays. Only VC0 with bubble flow control
// receives first-level acks.
manifest::SenderChannelCredits collect_sender_credits(
    const SenderCollectionContext& ctx, const SenderChannelIndex& index) {
    const bool counters = ctx.plan.vc_uses_counters(index.vc);
    const auto credit = [&](const char* array, const char* stream_name_pattern) -> manifest::CreditRef {
        if (counters) {
            return manifest::ArrayRef{.array = array, .index = index.compact};
        }
        return assigned_stream(ctx, fmt::format(fmt::runtime(stream_name_pattern), index.compact));
    };

    manifest::SenderChannelCredits credits{
        .completed = credit("credit_counters/to_sender_completion", "TO_SENDER_{}_PKTS_COMPLETED_ID"),
    };
    if (index.vc == 0 && ctx.vc0_bubble_flow_control) {
        credits.acked = credit("credit_counters/to_sender_ack", "TO_SENDER_{}_PKTS_ACKED_ID");
    }
    return credits;
}

manifest::SenderChannel collect_sender_channel(const SenderCollectionContext& ctx, const SenderChannelIndex& index) {
    const auto& erisc_builder = ctx.erisc_builder;
    const auto& config = erisc_builder.config;

    // Check that the compile-time arguments match the builder's info for whether the channel is serviced
    manifest::SenderChannel sender;
    for (uint32_t risc_id = 0; risc_id < ctx.named_ct_args_per_risc.size(); ++risc_id) {
        const bool serviced = erisc_builder.is_sender_channel_serviced(risc_id, index.compact);
        const auto name = fmt::format("IS_SENDER_CHANNEL_{}_SERVICED", index.compact);
        TT_FATAL(
            (get_named_arg(ctx.named_ct_args_per_risc[risc_id], name) != 0) == serviced,
            "Fabric manifest: ERISC{} {} is {}, but the builder has the channel {}",
            risc_id,
            name,
            get_named_arg(ctx.named_ct_args_per_risc[risc_id], name),
            serviced ? "serviced" : "not serviced");
        if (serviced) {
            sender.serviced_by.push_back(risc_id);
        }
    }

    // Get the producer for the channel
    sender.producer = collect_sender_producer(ctx, index);

    // Check that the compile-time arguments match the builder's info for whether the channel is an injection channel
    sender.is_injection_channel = erisc_builder.sender_channel_is_traffic_injection_channel_array.at(index.compact);
    check_named_arg(
        ctx.named_ct_args_per_risc,
        fmt::format("SENDER_CH_{}_IS_INJECTION", index.compact),
        sender.is_injection_channel ? 1 : 0);

    // Check that the compile-time arguments match the builder's info for the channel's ack NOC and command buffer
    const auto ack_noc = static_cast<uint32_t>(config.sender_channel_ack_noc_ids[index.compact]);
    const auto ack_cmd_buf = static_cast<uint32_t>(config.sender_channel_ack_cmd_buf_ids[index.compact]);
    check_named_arg(ctx.named_ct_args_per_risc, fmt::format("SENDER_CH_{}_ACK_NOC_ID", index.compact), ack_noc);
    check_named_arg(ctx.named_ct_args_per_risc, fmt::format("SENDER_CH_{}_ACK_CMD_BUF_ID", index.compact), ack_cmd_buf);
    sender.producer_credit_return = {
        .noc = static_cast<tt::tt_metal::NOC>(ack_noc),
        .cmd_buf = static_cast<manifest::NocCmdBuf>(ack_cmd_buf),
    };

    // Ring buffer info for the channel
    const size_t num_slots = ctx.allocator.get_sender_channel_number_of_slots(index.vc, index.channel);
    sender.ring_buffer = l1_region(
        ctx.allocator.get_sender_channel_base_address(index.vc, index.channel),
        num_slots * config.channel_buffer_size_bytes,
        "packet_ring",
        ctx.addresses_to_clear);
    sender.ring_buffer.num_elements = static_cast<uint32_t>(num_slots);
    sender.ring_buffer.size_per_element = static_cast<uint32_t>(config.channel_buffer_size_bytes);

    // Credits and free slots stream registers
    sender.free_slots =
        assigned_stream(ctx, fmt::format("SENDER_CHANNEL_{}_FREE_SLOTS_STREAM_ID", index.fabric_position));
    sender.credits = collect_sender_credits(ctx, index);

    // Sender channel and worker connection info
    const size_t conn_info_address = config.sender_channels_worker_conn_info_base_address[index.compact];
    check_named_arg(
        ctx.named_ct_args_per_risc,
        fmt::format("LOCAL_SENDER_CH_{}_CONN_INFO_ADDR", index.compact),
        static_cast<uint32_t>(conn_info_address));
    sender.control_info = {
        .connection = l1_region(
            config.sender_channels_connection_semaphore_address[index.compact],
            FabricEriscDatamoverConfig::field_size,
            "u32",
            ctx.addresses_to_clear),
        .conn_info = l1_region(
            conn_info_address,
            sizeof(EDMChannelWorkerLocationInfo),
            "struct:EDMChannelWorkerLocationInfo",
            ctx.addresses_to_clear),
    };

    // Only local workers actually update the buffer index semaphore as static connections do not close their connections 
    if (sender.producer.has_value() && std::holds_alternative<manifest::LocalWorker>(*sender.producer)) {
        sender.control_info.buffer_index_sem = l1_region(
            config.sender_channels_buffer_index_semaphore_address[index.compact],
            sizeof(SenderChannelProducerCursor),
            "struct:SenderChannelProducerCursor",
            ctx.addresses_to_clear);
    }

    // A serviced channel polls its free-slots register and, on registers, its credit registers.
    if (!sender.serviced_by.empty()) {
        const auto require_allocated = [&](const char* role, const manifest::StreamRef& stream) {
            TT_FATAL(
                stream.stream_id != k_unused_stream_id,
                "Fabric manifest: VC{} channel {} is serviced, but its {} stream register is not allocated",
                index.vc,
                index.channel,
                role);
        };
        require_allocated("free_slots", sender.free_slots);
        if (const auto* stream = std::get_if<manifest::StreamRef>(&sender.credits.completed)) {
            require_allocated("completed", *stream);
        }
        if (sender.credits.acked.has_value()) {
            if (const auto* stream = std::get_if<manifest::StreamRef>(&*sender.credits.acked)) {
                require_allocated("acked", *stream);
            }
        }
    }
    return sender;
}

// Every sender channel in the router's shape, indexed [vc][channel].
std::vector<std::vector<manifest::SenderChannel>> collect_senders(
    const FabricEriscDatamoverBuilder& erisc_builder,
    const RouterVcShape& vc_shape,
    const RouterLocation& location,
    const manifest::RouterShape& shape,
    const std::vector<NamedArgs>& named_ct_args_per_risc) {
    const auto* allocator =
        dynamic_cast<const FabricStaticSizedChannelsAllocator*>(erisc_builder.config.channel_allocator.get());
    TT_FATAL(allocator != nullptr, "Fabric manifest: the router's channel allocator is not statically sized");

    // Get the stream assignment for the router's mesh
    const auto& stream_assignment = builder_context().get_stream_assignment(erisc_builder.local_fabric_node_id.mesh_id);
    NamedArgs assigned_streams;
    for (const auto& [name, stream_id] : stream_assignment.named_args()) {
        assigned_streams.emplace(name, stream_id);
    }

    // Get the producer slots for the router's direction
    const builder::RouterProducerSlots producer_slots(
        builder::routing_direction_to_eth_direction(location.direction), vc_shape.sender_counts);
    const auto addresses_to_clear = builder_context().get_fabric_router_addresses_to_clear();

    // Context for sender channels on this router
    const SenderCollectionContext ctx{
        .erisc_builder = erisc_builder,
        .allocator = *allocator,
        .producer_slots = producer_slots,
        .named_ct_args_per_risc = named_ct_args_per_risc,
        .assigned_streams = assigned_streams,
        .plan = stream_assignment.plan(),
        .addresses_to_clear = addresses_to_clear,
        .vc0_bubble_flow_control = shape.vc0_bubble_flow_control,
    };

    // Collect the sender channels for each VC
    std::vector<std::vector<manifest::SenderChannel>> senders(builder_config::MAX_NUM_VCS);
    for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        for (uint32_t channel = 0; channel < vc_shape.sender_counts[vc]; ++channel) {
            senders[vc].push_back(collect_sender_channel(
                ctx,
                {.vc = vc,
                 .channel = channel,
                 .compact = vc_shape.flat_sender_id(vc, channel),
                 .fabric_position = stream_assignment.sender_flat_base(vc) + channel}));
        }
    }
    return senders;
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

    auto shape = collect_shape(erisc_builder, vc_shape, named_ct_args_per_risc);
    auto senders = collect_senders(erisc_builder, vc_shape, location, shape, named_ct_args_per_risc);
    return {
        .identity = collect_identity(location),
        .link = collect_link(erisc_builder, location, chip_facts),
        .shape = std::move(shape),
        .credit_counters = collect_credit_counters(erisc_builder, named_ct_args_per_risc),
        .channels = {.senders = std::move(senders)},
    };
}

}  // namespace tt::tt_fabric
