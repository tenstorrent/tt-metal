// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_collector.hpp"

#include <enchantum/enchantum.hpp>
#include <fmt/format.h>
#include <tt_stl/assert.hpp>
#include <tt_stl/overloaded.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>

#include <algorithm>
#include <array>
#include <numeric>
#include <span>
#include <variant>

#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp"
#include "tt_metal/fabric/builder/fabric_stream_assignment.hpp"
#include "tt_metal/fabric/builder/protected_domain_effect.hpp"
#include "tt_metal/fabric/builder/router_wiring_rules.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/fabric_router_builder.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_fields.hpp"
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
    // VC0's first-level acks exist for bubble flow control, so the manifest records the two as one fact.
    check_named_arg(named_ct_args_per_risc, "ENABLE_FIRST_LEVEL_ACK_VC0", shape.vc0_bubble_flow_control ? 1 : 0);

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

// The builder's NoC id, which every RISC must receive as `name`.
tt::tt_metal::NOC emitted_noc(
    const std::vector<NamedArgs>& named_ct_args_per_risc, const std::string& name, size_t noc) {
    check_named_arg(named_ct_args_per_risc, name, static_cast<uint32_t>(noc));
    return static_cast<tt::tt_metal::NOC>(noc);
}

// The builder's command buffer id, which every RISC must receive as `name`.
manifest::NocCmdBuf emitted_cmd_buf(
    const std::vector<NamedArgs>& named_ct_args_per_risc, const std::string& name, size_t cmd_buf) {
    check_named_arg(named_ct_args_per_risc, name, static_cast<uint32_t>(cmd_buf));
    return noc_cmd_buf(cmd_buf);
}

// The channel a per-channel field is read for. Zero for router and ERISC fields.
struct FieldIndex {
    uint32_t compact = 0;
    uint32_t fabric_position = 0;
};

// `arg`'s name, with the channel's index in its {}.
std::string arg_name(const manifest::NamedArg& arg, const FieldIndex& index) {
    switch (arg.index) {
        case manifest::ArgIndex::NONE: return std::string(arg.name);
        case manifest::ArgIndex::COMPACT: return fmt::format(fmt::runtime(arg.name), index.compact);
        case manifest::ArgIndex::FABRIC_POSITION: return fmt::format(fmt::runtime(arg.name), index.fabric_position);
    }
    TT_THROW("Fabric manifest: unknown argument index {}", static_cast<int>(arg.index));
}

// `field`, given the value of its source.
manifest::Field make_field(const ManifestRouterInputs& inputs, const manifest::RouterField& field, uint32_t arg) {
    const std::string_view key = field.key.value;
    const manifest::Kind& kind = field.kind.value;
    TT_FATAL(
        !std::holds_alternative<manifest::kind::Flag>(kind) || arg <= 1,
        "Fabric manifest: field {} is a flag, but is {}",
        key,
        arg);
    if (const auto* enum_kind = std::get_if<manifest::kind::Enum>(&kind)) {
        TT_FATAL(
            enum_kind->is_enumerator(arg),
            "Fabric manifest: field {} is {}, which is not a {}",
            key,
            arg,
            manifest::schema_name(enum_kind->element.type, enum_kind->element.size));
    }
    return {
        .key = key,
        .category = field.category.value,
        .kind = kind,
        .arg = arg,
        .cleared_by_host = std::holds_alternative<manifest::kind::L1Value>(kind) && is_cleared_by_host(inputs, arg),
    };
}

// The fields in `table`, for the channel at `index` when the table is per channel. `read_arg` reads a named argument
// by its name.
template <typename ReadArg>
std::vector<manifest::Field> collect_fields(
    const ManifestRouterInputs& inputs,
    std::span<const manifest::RouterField> table,
    const FieldIndex& index,
    const ReadArg& read_arg) {
    std::vector<manifest::Field> fields;
    for (const auto& field : table) {
        const uint32_t value = std::visit(
            ttsl::overloaded{
                [&](const manifest::NamedArg& arg) { return read_arg(arg_name(arg, index)); },
                [&](const manifest::BuilderMember& member) { return member.read(inputs.erisc_builder, index.compact); },
            },
            field.source);
        fields.push_back(make_field(inputs, field, value));
    }
    return fields;
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

// A receiver channel's indices. The kernel indexes every receiver table by the compact index.
struct ReceiverChannelIndex {
    uint32_t vc;
    uint32_t channel;
    uint32_t compact;
};

// What collecting a router's channels reads beyond its inputs.
struct ChannelContext {
    const ManifestRouterInputs& inputs;
    const FabricStaticSizedChannelsAllocator& allocator;
    const builder::RouterProducerSlots producer_slots;
    const bool is_2d_fabric;
    const bool speedy_vc0;
    // The router's VC0 senders other than the worker channel are carried by its tensix mux.
    const bool mux_mode;
    // The channel trimming profile the builder applied to this router, if any.
    const std::optional<ChannelTrimmingOverrides>& trimming;
    // The stream register ids the builder assigned, by compile-time argument name.
    const NamedArgs assigned_streams;
};

// The stream register the builder assigned under `name`, which every RISC must receive.
manifest::StreamRef assigned_stream(const ChannelContext& ctx, const std::string& name) {
    const uint32_t stream_id = get_named_arg(ctx.assigned_streams, name);
    check_named_arg(ctx.inputs.named_ct_args_per_risc, name, stream_id);
    return {.stream_id = stream_id, .schema = manifest::schema_name_of<uint32_t>()};
}

// A serviced channel polls this register, so it must be allocated.
void require_allocated(
    const char* kind, uint32_t vc, uint32_t channel, const char* role, const manifest::StreamRef& stream) {
    TT_FATAL(
        stream.stream_id != k_unused_stream_id,
        "Fabric manifest: VC{} {} channel {} is serviced, but its {} stream register is not allocated",
        vc,
        kind,
        channel,
        role);
}

// The kernel runs VC1's channel steps only under FABRIC_2D_VC1_SERVICED, and VC2's only under FABRIC_2D_VC2_SERVICED.
bool kernel_runs_vc(const ManifestRouterInputs& inputs, uint32_t vc) {
    switch (vc) {
        case 1: return inputs.kernel_defines.contains("FABRIC_2D_VC1_SERVICED");
        case 2: return inputs.kernel_defines.contains("FABRIC_2D_VC2_SERVICED");
        default: return true;
    }
}

// The ERISCs whose `serviced_name` argument is set, or none when the kernel does not run the VC's steps.
std::vector<uint32_t> collect_serviced_by(
    const ManifestRouterInputs& inputs, uint32_t vc, const std::string& serviced_name) {
    if (!kernel_runs_vc(inputs, vc)) {
        return {};
    }
    std::vector<uint32_t> serviced_by;
    for (uint32_t risc_id = 0; risc_id < inputs.named_ct_args_per_risc.size(); ++risc_id) {
        if (get_named_arg(inputs.named_ct_args_per_risc[risc_id], serviced_name) != 0) {
            serviced_by.push_back(risc_id);
        }
    }
    return serviced_by;
}

// Why the channel's step runs or not. An unserviced channel must be off for a reason the manifest knows, which it
// checks in the order the builder applies them: the kernel's VC gate, then mux mode, then the trimming profile.
manifest::ChannelStatus collect_status(
    const ChannelContext& ctx,
    const char* kind,
    uint32_t vc,
    uint32_t channel,
    const std::vector<uint32_t>& serviced_by,
    bool carried_by_mux,
    bool trimmed) {
    if (!serviced_by.empty()) {
        return manifest::ChannelStatus::ACTIVE;
    }
    if (!kernel_runs_vc(ctx.inputs, vc)) {
        return manifest::ChannelStatus::VC_NOT_SERVICED;
    }
    if (carried_by_mux) {
        return manifest::ChannelStatus::MUX;
    }
    if (trimmed) {
        return manifest::ChannelStatus::TRIMMED;
    }
    TT_THROW(
        "Fabric manifest: VC{} {} channel {} is serviced by no ERISC, for a reason the manifest does not know",
        vc,
        kind,
        channel);
}

// The worker channel is fed by the local worker, or in mux mode by the tensix mux. Any other channel is fed by a
// sibling router once one has connected to it, which makes the channel's connection static.
std::optional<manifest::SenderChannelProducer> collect_sender_producer(
    const ChannelContext& ctx, const SenderChannelIndex& index) {
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
        if (ctx.mux_mode && index.vc == 0) {
            return manifest::LocalTensixMux{};
        }
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
    const ChannelContext& ctx, const SenderChannelIndex& index, bool vc0_bubble_flow_control) {
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
    const ChannelContext& ctx, const SenderChannelIndex& index, bool vc0_bubble_flow_control) {
    const auto& inputs = ctx.inputs;
    const auto& args = inputs.named_ct_args_per_risc;
    const uint32_t c = index.compact;

    manifest::SenderChannel sender;
    sender.serviced_by = collect_serviced_by(inputs, index.vc, fmt::format("IS_SENDER_CHANNEL_{}_SERVICED", c));
    sender.status = collect_status(
        ctx,
        "sender",
        index.vc,
        index.channel,
        sender.serviced_by,
        ctx.mux_mode && index.vc == 0 && ctx.producer_slots.worker_channel(0) != index.channel,
        ctx.trimming.has_value() && !ctx.trimming->is_sender_channel_used(c));
    sender.producer = collect_sender_producer(ctx, index);
    sender.ring_buffer = ring_buffer_region(
        inputs,
        ctx.allocator.get_sender_channel_base_address(index.vc, index.channel),
        ctx.allocator.get_sender_channel_number_of_slots(index.vc, index.channel));
    sender.credits = collect_sender_credits(ctx, index, vc0_bubble_flow_control);
    sender.fields = collect_fields(
        inputs,
        manifest::k_sender_channel_fields,
        {.compact = c, .fabric_position = index.fabric_position},
        [&](const std::string& name) { return emitted_value(args, name); });

    // A serviced channel polls its credit registers, when its credits are on registers.
    if (!sender.serviced_by.empty()) {
        if (const auto* stream = std::get_if<manifest::StreamRef>(&sender.credits.completed)) {
            require_allocated("sender", index.vc, index.channel, "completed", *stream);
        }
        if (sender.credits.acked.has_value()) {
            if (const auto* stream = std::get_if<manifest::StreamRef>(&*sender.credits.acked)) {
                require_allocated("sender", index.vc, index.channel, "acked", *stream);
            }
        }
    }
    return sender;
}

// The kernel runs VC0's receiver step on channel 0 and VC1's on channel 1. VC2's runs on channel 2 when VC1 has
// senders and on channel 1 otherwise (VC2_RECEIVER_CHANNEL).
uint32_t kernel_receiver_channel(const ManifestRouterInputs& inputs, uint32_t vc) {
    if (vc < 2) {
        return vc;
    }
    return inputs.vc_shape.sender_counts[1] > 0 ? 2 : 1;
}

// The VC whose downstream edges the kernel gives the receiver's step: its own, except that receiver 0 gets VC1's
// under FABRIC_2D_VC0_CROSSOVER_TO_VC1. The speedy steps, VC0's in speedy mode and VC2's always, are given none.
std::optional<uint32_t> collect_forwards_on(const ChannelContext& ctx, uint32_t vc, bool serviced) {
    if (!serviced || vc == 2 || (vc == 0 && ctx.speedy_vc0)) {
        return std::nullopt;
    }
    if (vc == 0 && ctx.inputs.kernel_defines.contains("FABRIC_2D_VC0_CROSSOVER_TO_VC1")) {
        return 1;
    }
    return vc;
}

manifest::ReceiverChannel collect_receiver_channel(const ChannelContext& ctx, const ReceiverChannelIndex& index) {
    const auto& inputs = ctx.inputs;
    const auto& config = inputs.erisc_builder.config;
    const auto& args = inputs.named_ct_args_per_risc;
    const uint32_t c = index.compact;

    manifest::ReceiverChannel receiver;
    receiver.serviced_by = collect_serviced_by(inputs, index.vc, fmt::format("IS_RECEIVER_CHANNEL_{}_SERVICED", c));
    receiver.status = collect_status(
        ctx,
        "receiver",
        index.vc,
        index.channel,
        receiver.serviced_by,
        /*carried_by_mux=*/false,
        ctx.trimming.has_value() && !ctx.trimming->is_receiver_channel_data_forwarded(c));
    const bool serviced = !receiver.serviced_by.empty();
    TT_FATAL(
        !serviced || c == kernel_receiver_channel(inputs, index.vc),
        "Fabric manifest: VC{} receiver channel {} is compact channel {}, but the kernel runs VC{}'s receiver on "
        "channel {}",
        index.vc,
        index.channel,
        c,
        index.vc,
        kernel_receiver_channel(inputs, index.vc));
    receiver.forwards_on = collect_forwards_on(ctx, index.vc, serviced);

    receiver.forwarding_disabled = emitted_flag(args, fmt::format("DISABLE_RX_CH{}_FORWARDING", c));
    // 1D kernels get no ingress arguments and treat every receiver as not an ingress.
    receiver.intermesh_ingress =
        ctx.is_2d_fabric && emitted_flag(args, fmt::format("IS_RECEIVER_CHANNEL_{}_INTERMESH_INGRESS", c));

    receiver.forward_noc = {
        .noc = emitted_noc(args, fmt::format("RX_CH_{}_FWD_NOC_ID", c), config.receiver_channel_forwarding_noc_ids[c]),
        .data_cmd_buf = emitted_cmd_buf(
            args,
            fmt::format("RX_CH_{}_FWD_DATA_CMD_BUF_ID", c),
            config.receiver_channel_forwarding_data_cmd_buf_ids[c]),
        .sync_cmd_buf = emitted_cmd_buf(
            args,
            fmt::format("RX_CH_{}_FWD_SYNC_CMD_BUF_ID", c),
            config.receiver_channel_forwarding_sync_cmd_buf_ids[c]),
    };
    receiver.local_write_noc = {
        .noc = emitted_noc(
            args, fmt::format("RX_CH_{}_LOCAL_WRITE_NOC_ID", c), config.receiver_channel_local_write_noc_ids[c]),
        .cmd_buf = emitted_cmd_buf(
            args,
            fmt::format("RX_CH_{}_LOCAL_WRITE_CMD_BUF_ID", c),
            config.receiver_channel_local_write_cmd_buf_ids[c]),
    };

    receiver.ring_buffer = ring_buffer_region(
        inputs,
        ctx.allocator.get_receiver_channel_base_address(index.vc, index.channel),
        ctx.allocator.get_receiver_channel_number_of_slots(index.vc, index.channel));
    receiver.pkts_sent = assigned_stream(ctx, fmt::format("TO_RECEIVER_{}_PKTS_SENT_ID", c));
    if (index.vc == 2) {
        receiver.free_slots = assigned_stream(ctx, "VC2_RECEIVER_FREE_SLOTS_STREAM_ID");
    }

    if (serviced) {
        require_allocated("receiver", index.vc, index.channel, "pkts_sent", receiver.pkts_sent);
        if (receiver.free_slots.has_value()) {
            require_allocated("receiver", index.vc, index.channel, "free_slots", *receiver.free_slots);
        }
    }
    return receiver;
}

ChannelContext make_channel_context(const ManifestRouterInputs& inputs) {
    const auto& config = inputs.erisc_builder.config;
    const auto* allocator = dynamic_cast<const FabricStaticSizedChannelsAllocator*>(config.channel_allocator.get());
    TT_FATAL(allocator != nullptr, "Fabric manifest: the router's channel allocator is not statically sized");
    check_named_arg(
        inputs.named_ct_args_per_risc, "CHANNEL_BUFFER_SIZE", static_cast<uint32_t>(config.channel_buffer_size_bytes));

    const auto assigned_streams = inputs.stream_assignment.named_args();
    return {
        .inputs = inputs,
        .allocator = *allocator,
        .producer_slots = builder::RouterProducerSlots(
            builder::routing_direction_to_eth_direction(inputs.location.direction), inputs.vc_shape.sender_counts),
        .is_2d_fabric = emitted_flag(inputs.named_ct_args_per_risc, "IS_2D_FABRIC"),
        .speedy_vc0 = emitted_flag(inputs.named_ct_args_per_risc, "ENABLE_SPEEDY_VC0"),
        .mux_mode = emitted_flag(inputs.named_ct_args_per_risc, "FABRIC_TENSIX_EXTENSION_MUX_MODE"),
        .trimming = inputs.erisc_builder.get_channel_trimming_overrides(),
        .assigned_streams = NamedArgs(assigned_streams.begin(), assigned_streams.end()),
    };
}

// Every sender and receiver channel in the router's shape, each indexed [vc][channel].
manifest::Channels collect_channels(const ChannelContext& ctx, const manifest::RouterShape& shape) {
    const auto& inputs = ctx.inputs;
    manifest::Channels channels;
    channels.senders.resize(builder_config::MAX_NUM_VCS);
    channels.receivers.resize(builder_config::MAX_NUM_VCS);
    for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        for (uint32_t channel = 0; channel < inputs.vc_shape.sender_counts[vc]; ++channel) {
            const SenderChannelIndex index{
                .vc = vc,
                .channel = channel,
                .compact = inputs.vc_shape.flat_sender_id(vc, channel),
                .fabric_position = inputs.stream_assignment.sender_flat_base(vc) + channel,
            };
            channels.senders[vc].push_back(collect_sender_channel(ctx, index, shape.vc0_bubble_flow_control));
        }
        for (uint32_t channel = 0; channel < inputs.vc_shape.receiver_counts[vc]; ++channel) {
            const ReceiverChannelIndex index{
                .vc = vc,
                .channel = channel,
                .compact = inputs.vc_shape.flat_receiver_id(vc, channel),
            };
            channels.receivers[vc].push_back(collect_receiver_channel(ctx, index));
        }
    }
    return channels;
}

// Receivers forwarding on the same VC share its edges, so they must forward on the same NoC. Returns, per VC,
// whether a serviced receiver forwards on it.
std::array<bool, builder_config::MAX_NUM_VCS> check_forwarding_receivers(const manifest::Channels& channels) {
    std::array<const manifest::ReceiverChannel*, builder_config::MAX_NUM_VCS> first_on_vc{};
    std::array<bool, builder_config::MAX_NUM_VCS> forwarded_on{};
    for (const auto& receivers : channels.receivers) {
        for (const auto& receiver : receivers) {
            if (!receiver.forwards_on.has_value()) {
                continue;
            }
            const uint32_t vc = *receiver.forwards_on;
            forwarded_on.at(vc) = true;
            auto& first = first_on_vc.at(vc);
            if (first == nullptr) {
                first = &receiver;
                continue;
            }
            TT_FATAL(
                first->forward_noc.noc == receiver.forward_noc.noc,
                "Fabric manifest: two receivers forward on VC{}'s edges, on NoC {} and NoC {}",
                vc,
                static_cast<uint32_t>(first->forward_noc.noc),
                static_cast<uint32_t>(receiver.forward_noc.noc));
        }
    }
    return forwarded_on;
}

// The router's edges on `vc`, ordered by edge. The kernel takes edge n's free-slots register from
// VC<vc>_FREE_SLOTS_FROM_DOWNSTREAM_EDGE_<n>_STREAM_ID, and its teardown semaphore by the edge's rank among the
// router's edges, VC0's before VC1's. It builds VC1's edges only under FABRIC_2D_VC1_ACTIVE, and gives VC2 none.
std::vector<manifest::DownstreamEdge> collect_downstream_edges(
    const ChannelContext& ctx, uint32_t vc, bool forwarded_on) {
    const auto& inputs = ctx.inputs;
    const auto& erisc_builder = inputs.erisc_builder;
    const auto& config = erisc_builder.config;
    const auto& adapter = *erisc_builder.receiver_channel_to_downstream_adapter;
    const auto& connections = adapter.get_downstream_connections(vc);
    // The kernel never forwards on VC2.
    if (vc == 2) {
        return {};
    }
    check_named_arg(
        inputs.named_ct_args_per_risc,
        fmt::format("NUM_DOWNSTREAM_SENDERS_VC{}", vc),
        static_cast<uint32_t>(connections.size()));
    TT_FATAL(
        vc == 0 || connections.empty() || inputs.kernel_defines.contains("FABRIC_2D_VC1_ACTIVE"),
        "Fabric manifest: the router has VC1 downstream edges, but no FABRIC_2D_VC1_ACTIVE to build them");

    const auto my_direction = builder::routing_direction_to_eth_direction(inputs.location.direction);
    std::vector<manifest::DownstreamEdge> edges;
    uint32_t mask = 0;
    for (const auto& [direction, core] : connections) {
        const uint32_t slot = ctx.is_2d_fabric ? get_receiver_channel_compact_index(my_direction, direction) : 0;
        TT_FATAL((mask & (1u << slot)) == 0, "Fabric manifest: VC{} has two downstream edges in slot {}", vc, slot);
        mask |= 1u << slot;

        std::optional<uint32_t> landing_compact;
        if (ctx.is_2d_fabric) {
            const auto id = adapter.get_downstream_sender_channel_id(vc, slot);
            TT_FATAL(id.has_value(), "Fabric manifest: VC{} edge {} has no landing channel", vc, slot + 1);
            landing_compact = static_cast<uint32_t>(*id);
        }
        edges.push_back({
            .edge = slot + 1,
            .target = {.direction = direction},
            .landing_channel =
                builder::get_downstream_sender_channel_for_vc(ctx.is_2d_fabric, vc, my_direction, direction),
            .landing_compact = landing_compact,
            .core = core,
            .free_slots =
                assigned_stream(ctx, fmt::format("VC{}_FREE_SLOTS_FROM_DOWNSTREAM_EDGE_{}_STREAM_ID", vc, slot + 1)),
        });
    }
    TT_FATAL(
        mask == adapter.get_downstream_edm_mask_for_vc(vc),
        "Fabric manifest: VC{}'s downstream edges are in slots {:#x}, but the kernel is given {:#x}",
        vc,
        mask,
        adapter.get_downstream_edm_mask_for_vc(vc));
    std::ranges::sort(edges, {}, &manifest::DownstreamEdge::edge);

    const size_t teardown_base = vc == 1 ? adapter.get_downstream_connections(0).size() : 0;
    for (size_t rank = 0; rank < edges.size(); ++rank) {
        auto& edge = edges[rank];
        const size_t teardown_index = teardown_base + rank;
        TT_FATAL(
            teardown_index < config.num_fwd_paths,
            "Fabric manifest: VC{} edge {} takes teardown semaphore {}, but the kernel has {}",
            vc,
            edge.edge,
            teardown_index,
            config.num_fwd_paths);
        // The kernel takes the address from its runtime args.
        const size_t teardown_address =
            config.receiver_channels_downstream_teardown_semaphore_address.at(teardown_index);
        TT_FATAL(
            erisc_builder.receiver_channels_downstream_teardown_semaphore_id.at(teardown_index) == teardown_address,
            "Fabric manifest: teardown semaphore {}'s runtime arg does not hold its address",
            teardown_index);
        edge.teardown_sem = l1_value<uint32_t>(inputs, teardown_address);
        TT_FATAL(
            !forwarded_on || edge.free_slots.stream_id != k_unused_stream_id,
            "Fabric manifest: a receiver forwards on VC{} edge {}, but its free_slots stream register is not allocated",
            vc,
            edge.edge);
    }
    return edges;
}

std::vector<std::vector<manifest::DownstreamEdge>> collect_router_edges(
    const ChannelContext& ctx, const manifest::Channels& channels) {
    const auto forwarded_on = check_forwarding_receivers(channels);
    std::vector<std::vector<manifest::DownstreamEdge>> edges;
    for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        edges.push_back(collect_downstream_edges(ctx, vc, forwarded_on[vc]));
    }
    return edges;
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
    const auto ctx = make_channel_context(inputs);
    auto channels = collect_channels(ctx, shape);
    auto edges = collect_router_edges(ctx, channels);

    const auto& args = inputs.named_ct_args_per_risc;
    auto fields = collect_fields(
        inputs, manifest::k_router_fields, {}, [&](const std::string& name) { return emitted_value(args, name); });
    std::vector<std::vector<manifest::Field>> erisc_fields;
    for (const auto& risc_args : args) {
        erisc_fields.push_back(collect_fields(inputs, manifest::k_erisc_fields, {}, [&](const std::string& name) {
            return get_named_arg(risc_args, name);
        }));
    }
    return {
        .identity = collect_identity(inputs.location),
        .link = collect_link(inputs),
        .shape = shape,
        .credit_counters = collect_credit_counters(inputs),
        .channels = std::move(channels),
        .intra_chip_downstream_edges = std::move(edges),
        .fields = std::move(fields),
        .erisc_fields = std::move(erisc_fields),
    };
}

}  // namespace tt::tt_fabric
