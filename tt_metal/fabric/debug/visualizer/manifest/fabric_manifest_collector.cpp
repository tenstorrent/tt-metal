// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_collector.hpp"

#include <enchantum/enchantum.hpp>
#include <fmt/format.h>
#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>

#include <algorithm>
#include <array>
#include <numeric>

#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp"
#include "tt_metal/fabric/builder/fabric_stream_assignment.hpp"
#include "tt_metal/fabric/builder/protected_domain_effect.hpp"
#include "tt_metal/fabric/builder/router_wiring_rules.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/fabric_router_builder.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_struct_layouts.hpp"

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

bool is_cleared_by_host(const ManifestRouterInputs& inputs, size_t address) {
    return std::ranges::find(inputs.addresses_to_clear, address) != inputs.addresses_to_clear.end();
}

// An array of T filling `size` bytes at `address`.
template <typename T>
manifest::L1Region l1_array(const ManifestRouterInputs& inputs, size_t address, size_t size) {
    TT_FATAL(
        size % sizeof(T) == 0,
        "Fabric manifest: array at {:#x} is {} bytes, not a whole number of {}-byte elements",
        address,
        size,
        sizeof(T));
    return {
        .address = static_cast<uint32_t>(address),
        .size = static_cast<uint32_t>(size),
        .num_elements = static_cast<uint32_t>(size / sizeof(T)),
        .size_per_element = static_cast<uint32_t>(sizeof(T)),
        .schema = manifest::schema_name_of<T>(),
        .cleared_by_host = is_cleared_by_host(inputs, address),
    };
}

// A credit counter array's base address and the compile-time argument that passes it to the kernel.
struct CounterBase {
    const char* ct_arg_name;
    size_t address;
};

// Return the router's four L1 credit counter arrays. The kernel sends each pair to the peer in one packet, so it
// relies on the arrays being back to back and the same size; the size comes from that spacing.
manifest::L1CreditCounters collect_credit_counters(const ManifestRouterInputs& inputs) {
    const auto& config = inputs.erisc_builder.config;
    // In L1 order
    const std::array<CounterBase, 4> bases = {{
        {"TO_SENDER_REMOTE_ACK_COUNTERS_BASE_ADDR", config.to_sender_channel_remote_ack_counters_base_addr},
        {"TO_SENDER_REMOTE_COMPLETION_COUNTERS_BASE_ADDR",
         config.to_sender_channel_remote_completion_counters_base_addr},
        {"LOCAL_RECEIVER_ACK_COUNTERS_BASE_ADDR", config.receiver_channel_remote_ack_counters_base_addr},
        {"LOCAL_RECEIVER_COMPLETION_COUNTERS_BASE_ADDR", config.receiver_channel_remote_completion_counters_base_addr},
    }};

    for (const auto& base : bases) {
        check_named_arg(inputs.named_ct_args_per_risc, base.ct_arg_name, static_cast<uint32_t>(base.address));
    }

    const size_t size = bases[1].address - bases[0].address;
    for (size_t i = 1; i < bases.size(); ++i) {
        TT_FATAL(
            bases[i].address > bases[i - 1].address && bases[i].address - bases[i - 1].address == size,
            "Fabric manifest: credit counter array {} at {:#x} does not follow {} at {:#x} as a {}-byte array",
            bases[i].ct_arg_name,
            bases[i].address,
            bases[i - 1].ct_arg_name,
            bases[i - 1].address,
            size);
    }

    return {
        .to_sender_ack = l1_array<uint32_t>(inputs, bases[0].address, size),
        .to_sender_completion = l1_array<uint32_t>(inputs, bases[1].address, size),
        .receiver_ack = l1_array<uint32_t>(inputs, bases[2].address, size),
        .receiver_completion = l1_array<uint32_t>(inputs, bases[3].address, size),
    };
}

// Every RISC's VC*_USES_COUNTER_CREDITS arguments must match the mesh's credit plan, which the writer serializes as
// the mesh's credit_transport.
void check_credit_transport_args(const ManifestRouterInputs& inputs) {
    const auto& plan = inputs.stream_assignment.plan();
    for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        check_named_arg(
            inputs.named_ct_args_per_risc,
            fmt::format("VC{}_USES_COUNTER_CREDITS", vc),
            plan.vc_uses_counters(vc) ? 1 : 0);
    }
}

// One T at `address`.
template <typename T>
manifest::L1Region l1_value(const ManifestRouterInputs& inputs, size_t address) {
    return {
        .address = static_cast<uint32_t>(address),
        .size = static_cast<uint32_t>(sizeof(T)),
        .schema = manifest::schema_name_of<T>(),
        .cleared_by_host = is_cleared_by_host(inputs, address),
    };
}

// A channel's packet slots, each channel_buffer_size_bytes. No struct describes a slot: it holds a packet.
manifest::L1Region ring_buffer_region(const ManifestRouterInputs& inputs, size_t address, size_t num_slots) {
    const size_t slot_size = inputs.erisc_builder.config.channel_buffer_size_bytes;
    return {
        .address = static_cast<uint32_t>(address),
        .size = static_cast<uint32_t>(num_slots * slot_size),
        .num_elements = static_cast<uint32_t>(num_slots),
        .size_per_element = static_cast<uint32_t>(slot_size),
        .schema = "packet_ring",
        .cleared_by_host = is_cleared_by_host(inputs, address),
    };
}

manifest::NocCmdBuf noc_cmd_buf(size_t cmd_buf) {
    const auto value = enchantum::cast<manifest::NocCmdBuf>(static_cast<uint32_t>(cmd_buf));
    TT_FATAL(value.has_value(), "Fabric manifest: {} is not a NoC command buffer", cmd_buf);
    return *value;
}

// A sender channel's indices. The kernel indexes its per-channel tables by the compact index, over this router's
// own counts, except the free-slots stream table, which it indexes by the fabric position, over the fabric's
// maximum counts (StreamAssignment).
struct SenderChannelIndex {
    uint32_t vc;
    uint32_t channel;
    uint32_t compact;
    uint32_t fabric_position;
};

// What collecting a router's sender channels reads beyond its inputs.
struct SenderContext {
    const ManifestRouterInputs& inputs;
    const FabricStaticSizedChannelsAllocator& allocator;
    const builder::RouterProducerSlots producer_slots;
    const bool is_2d_fabric;
    // The stream register ids the builder assigned, by compile-time argument name.
    const NamedArgs assigned_streams;
};

// The stream register the builder assigned under `name`, which every RISC must receive.
manifest::StreamRef assigned_stream(const SenderContext& ctx, const std::string& name) {
    const uint32_t stream_id = get_named_arg(ctx.assigned_streams, name);
    check_named_arg(ctx.inputs.named_ct_args_per_risc, name, stream_id);
    return {.stream_id = stream_id, .schema = manifest::schema_name_of<uint32_t>()};
}

// A serviced channel polls this register, so it must be allocated.
void require_allocated(const SenderChannelIndex& index, const char* role, const manifest::StreamRef& stream) {
    TT_FATAL(
        stream.stream_id != k_unused_stream_id,
        "Fabric manifest: VC{} sender channel {} is serviced, but its {} stream register is not allocated",
        index.vc,
        index.channel,
        role);
}

// The ERISCs whose IS_SENDER_CHANNEL_<c>_SERVICED is set. A VC1 channel is serviced by none of them unless the kernel
// defines FABRIC_2D_VC1_SERVICED, since it runs VC1's channel steps only then.
std::vector<uint32_t> collect_sender_serviced_by(const ManifestRouterInputs& inputs, const SenderChannelIndex& index) {
    if (index.vc == 1 && !inputs.kernel_defines.contains("FABRIC_2D_VC1_SERVICED")) {
        return {};
    }
    const auto name = fmt::format("IS_SENDER_CHANNEL_{}_SERVICED", index.compact);
    std::vector<uint32_t> serviced_by;
    for (uint32_t risc_id = 0; risc_id < inputs.named_ct_args_per_risc.size(); ++risc_id) {
        if (get_named_arg(inputs.named_ct_args_per_risc[risc_id], name) != 0) {
            serviced_by.push_back(risc_id);
        }
    }
    return serviced_by;
}

// The worker channel is fed by the local worker. Any other channel is fed by a sibling router once one has connected
// to it, which makes the channel's connection static. In 2D that sibling is the one whose producer slot the channel
// is. A 1D router is connected to one sibling, in both directions, so the sibling feeding it is the one it forwards
// to.
std::optional<manifest::SenderChannelProducer> collect_sender_producer(
    const SenderContext& ctx, const SenderChannelIndex& index) {
    const auto& erisc_builder = ctx.inputs.erisc_builder;
    const bool static_connection = erisc_builder.sender_channel_connection_liveness_check_disable_array[index.compact];
    check_named_arg(
        ctx.inputs.named_ct_args_per_risc,
        fmt::format("SENDER_CH_{}_WAIT_STATIC_CONNECTION", index.compact),
        static_connection ? 1 : 0);

    if (ctx.producer_slots.worker_channel(index.vc) == index.channel) {
        TT_FATAL(
            !static_connection,
            "Fabric manifest: VC{} channel {} is the worker channel, but a router connected to it",
            index.vc,
            index.channel);
        return manifest::LocalWorker{};
    }
    if (!static_connection) {
        return std::nullopt;
    }

    if (ctx.is_2d_fabric) {
        const auto producer = ctx.producer_slots.producer_at(index.vc, index.channel);
        TT_FATAL(
            producer.has_value(),
            "Fabric manifest: a router connected to VC{} channel {}, which is not a producer slot",
            index.vc,
            index.channel);
        return manifest::SiblingRouterRef{.direction = *producer};
    }
    const auto& connections =
        erisc_builder.receiver_channel_to_downstream_adapter->get_downstream_connections(index.vc);
    TT_FATAL(
        connections.size() == 1,
        "Fabric manifest: a router connected to 1D VC{} channel {}, but this router forwards to {} routers on VC{}",
        index.vc,
        index.channel,
        connections.size(),
        index.vc);
    const auto sibling = connections.front().first;
    const auto landing_channel = builder::get_downstream_sender_channel_for_vc(
        false, index.vc, sibling, builder::routing_direction_to_eth_direction(ctx.inputs.location.direction));
    TT_FATAL(
        index.channel == landing_channel,
        "Fabric manifest: a router connected to 1D VC{} channel {}, but 1D routers connect to channel {}",
        index.vc,
        index.channel,
        landing_channel);
    return manifest::SiblingRouterRef{.direction = sibling};
}

// Stream-backed credits are the TO_SENDER_<c>_PKTS_* registers, which the kernel looks up by compact index.
// Counter-backed credits are the channel's element of the to_sender counter arrays. Only VC0 with bubble flow control
// gets first-level acks.
manifest::SenderChannelCredits collect_sender_credits(
    const SenderContext& ctx, const SenderChannelIndex& index, bool vc0_bubble_flow_control) {
    const bool counters = ctx.inputs.stream_assignment.plan().vc_uses_counters(index.vc);
    const auto credit = [&](manifest::CreditCounterArray array, const char* stream_name) -> manifest::CreditRef {
        if (counters) {
            return manifest::CounterRef{.array = array, .index = index.compact};
        }
        return assigned_stream(ctx, fmt::format(fmt::runtime(stream_name), index.compact));
    };

    manifest::SenderChannelCredits credits{
        .completed = credit(manifest::CreditCounterArray::TO_SENDER_COMPLETION, "TO_SENDER_{}_PKTS_COMPLETED_ID"),
    };
    if (index.vc == 0 && vc0_bubble_flow_control) {
        credits.acked = credit(manifest::CreditCounterArray::TO_SENDER_ACK, "TO_SENDER_{}_PKTS_ACKED_ID");
    }
    return credits;
}

manifest::SenderChannel collect_sender_channel(
    const SenderContext& ctx, const SenderChannelIndex& index, bool vc0_bubble_flow_control) {
    const auto& inputs = ctx.inputs;
    const auto& erisc_builder = inputs.erisc_builder;
    const auto& config = erisc_builder.config;
    const auto& args = inputs.named_ct_args_per_risc;
    const uint32_t c = index.compact;

    manifest::SenderChannel sender;
    sender.serviced_by = collect_sender_serviced_by(inputs, index);
    sender.producer = collect_sender_producer(ctx, index);

    sender.is_injection_channel = erisc_builder.sender_channel_is_traffic_injection_channel_array.at(c);
    check_named_arg(args, fmt::format("SENDER_CH_{}_IS_INJECTION", c), sender.is_injection_channel ? 1 : 0);

    const auto ack_noc = static_cast<uint32_t>(config.sender_channel_ack_noc_ids[c]);
    const auto ack_cmd_buf = config.sender_channel_ack_cmd_buf_ids[c];
    check_named_arg(args, fmt::format("SENDER_CH_{}_ACK_NOC_ID", c), ack_noc);
    check_named_arg(args, fmt::format("SENDER_CH_{}_ACK_CMD_BUF_ID", c), static_cast<uint32_t>(ack_cmd_buf));
    sender.producer_credit_return = {
        .noc = static_cast<tt::tt_metal::NOC>(ack_noc),
        .cmd_buf = noc_cmd_buf(ack_cmd_buf),
    };

    sender.ring_buffer = ring_buffer_region(
        inputs,
        ctx.allocator.get_sender_channel_base_address(index.vc, index.channel),
        ctx.allocator.get_sender_channel_number_of_slots(index.vc, index.channel));
    sender.free_slots =
        assigned_stream(ctx, fmt::format("SENDER_CHANNEL_{}_FREE_SLOTS_STREAM_ID", index.fabric_position));
    sender.credits = collect_sender_credits(ctx, index, vc0_bubble_flow_control);

    const size_t conn_info_address = config.sender_channels_worker_conn_info_base_address[c];
    check_named_arg(
        args, fmt::format("LOCAL_SENDER_CH_{}_CONN_INFO_ADDR", c), static_cast<uint32_t>(conn_info_address));
    // The kernel takes these two addresses from its runtime args, not its compile-time args.
    const size_t connection_address = config.sender_channels_connection_semaphore_address[c];
    const size_t buffer_index_sem_address = config.sender_channels_buffer_index_semaphore_address[c];
    TT_FATAL(
        erisc_builder.sender_channels_connection_semaphore_id[c] == connection_address &&
            erisc_builder.sender_channels_buffer_index_semaphore_id[c] == buffer_index_sem_address,
        "Fabric manifest: sender channel {}'s runtime args do not hold its semaphore addresses",
        c);
    sender.control_info = {
        .connection = l1_value<uint32_t>(inputs, connection_address),
        .conn_info = l1_value<EDMChannelWorkerLocationInfo>(inputs, conn_info_address),
        .buffer_index_sem = l1_value<SenderChannelProducerCursor>(inputs, buffer_index_sem_address),
    };

    // A serviced channel polls its free-slots register and, when its credits are on registers, its credit registers.
    if (!sender.serviced_by.empty()) {
        require_allocated(index, "free_slots", sender.free_slots);
        if (const auto* stream = std::get_if<manifest::StreamRef>(&sender.credits.completed)) {
            require_allocated(index, "completed", *stream);
        }
        if (sender.credits.acked.has_value()) {
            if (const auto* stream = std::get_if<manifest::StreamRef>(&*sender.credits.acked)) {
                require_allocated(index, "acked", *stream);
            }
        }
    }
    return sender;
}

// Every sender channel in the router's shape, indexed [vc][channel].
std::vector<std::vector<manifest::SenderChannel>> collect_senders(
    const ManifestRouterInputs& inputs, const manifest::RouterShape& shape) {
    const auto& config = inputs.erisc_builder.config;
    const auto* allocator = dynamic_cast<const FabricStaticSizedChannelsAllocator*>(config.channel_allocator.get());
    TT_FATAL(allocator != nullptr, "Fabric manifest: the router's channel allocator is not statically sized");
    check_named_arg(
        inputs.named_ct_args_per_risc, "CHANNEL_BUFFER_SIZE", static_cast<uint32_t>(config.channel_buffer_size_bytes));

    const auto assigned_streams = inputs.stream_assignment.named_args();
    const SenderContext ctx{
        .inputs = inputs,
        .allocator = *allocator,
        .producer_slots = builder::RouterProducerSlots(
            builder::routing_direction_to_eth_direction(inputs.location.direction), inputs.vc_shape.sender_counts),
        .is_2d_fabric = emitted_flag(inputs.named_ct_args_per_risc, "IS_2D_FABRIC"),
        .assigned_streams = NamedArgs(assigned_streams.begin(), assigned_streams.end()),
    };

    std::vector<std::vector<manifest::SenderChannel>> senders(builder_config::MAX_NUM_VCS);
    for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        for (uint32_t channel = 0; channel < inputs.vc_shape.sender_counts[vc]; ++channel) {
            const SenderChannelIndex index{
                .vc = vc,
                .channel = channel,
                .compact = inputs.vc_shape.flat_sender_id(vc, channel),
                .fabric_position = inputs.stream_assignment.sender_flat_base(vc) + channel,
            };
            senders[vc].push_back(collect_sender_channel(ctx, index, shape.vc0_bubble_flow_control));
        }
    }
    return senders;
}

}  // namespace

manifest::Router collect_manifest_router(const ManifestRouterInputs& inputs) {
    TT_FATAL(
        inputs.named_ct_args_per_risc.size() == inputs.erisc_builder.get_configured_risc_count(),
        "Fabric manifest: got compile-time arguments for {} RISCs, but the router runs {}",
        inputs.named_ct_args_per_risc.size(),
        inputs.erisc_builder.get_configured_risc_count());
    check_credit_transport_args(inputs);

    auto shape = collect_shape(inputs);
    auto senders = collect_senders(inputs, shape);
    return {
        .identity = collect_identity(inputs.location),
        .link = collect_link(inputs),
        .shape = shape,
        .credit_counters = collect_credit_counters(inputs),
        .channels = {.senders = std::move(senders)},
    };
}

}  // namespace tt::tt_fabric
