// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_collector.hpp"

#include <fmt/format.h>
#include <tt_stl/assert.hpp>
#include <tt_stl/overloaded.hpp>
#include <algorithm>
#include <array>
#include <numeric>
#include <span>
#include <variant>

#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp"
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

// Return what the router's location and chip-wide facts know about its link.
manifest::EthLink collect_link(const ManifestRouterInputs& inputs) {
    const auto& location = inputs.location;
    const auto& capability = inputs.chip_facts.per_direction_capabilities.at(location.direction);
    TT_FATAL(
        capability.has_value(),
        "Fabric manifest: router on channel {} has no classified edge in the direction it faces",
        location.eth_chan);
    return {
        .edge_capability = *capability,
        .is_dispatch_link = location.is_dispatch_link,
    };
}

// Return the RouterShape. The kernel is fed its sender counts and its ERISC count, but only the total of its
// receivers, so the receivers per VC and the number of VCs come from the VC shape.
manifest::RouterShape collect_shape(const ManifestRouterInputs& inputs) {
    const auto& args = inputs.kernel.named_ct_args;
    manifest::RouterShape shape{
        .num_vcs = inputs.vc_shape.num_vcs,
        .receivers_per_vc = inputs.vc_shape.receiver_counts,
        .num_active_eriscs = emitted_value(args, "NUM_ACTIVE_ERISCS"),
        .channel_trimming_overrides_applied = inputs.erisc_builder.get_channel_trimming_overrides().has_value(),
        .vc0_bubble_flow_control = emitted_flag(args, "ENABLE_DEADLOCK_AVOIDANCE"),
    };
    for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        shape.senders_per_vc[vc] = emitted_value(args, fmt::format("ACTUAL_VC{}_SENDER_CHANNELS", vc));
    }
    // Otherwise the manifest would list receivers the kernel does not loop over, or leave out ones it does.
    const uint32_t num_receivers = emitted_value(args, "NUM_RECEIVER_CHANNELS");
    const uint32_t listed = std::accumulate(shape.receivers_per_vc.begin(), shape.receivers_per_vc.end(), 0u);
    TT_FATAL(
        listed == num_receivers,
        "Fabric manifest: the VC shape has {} receiver channels, but the kernel runs {}",
        listed,
        num_receivers);
    return shape;
}

bool is_cleared_by_host(const ManifestRouterInputs& inputs, size_t address) {
    return std::ranges::find(inputs.addresses_to_clear, address) != inputs.addresses_to_clear.end();
}

// An array of T filling `size` bytes at `address`.
template <typename T>
manifest::content::L1 l1_array(const ManifestRouterInputs& inputs, size_t address, size_t size) {
    TT_FATAL(
        size % sizeof(T) == 0,
        "Fabric manifest: array at {:#x} is {} bytes, not a whole number of {}-byte elements",
        address,
        size,
        sizeof(T));
    return {
        .address = static_cast<uint32_t>(address),
        .type = layout::array_of<T>(static_cast<uint32_t>(size / sizeof(T))),
        .cleared_by_host = is_cleared_by_host(inputs, address),
    };
}

// A credit counter array's base address and the compile-time argument that passes it to the kernel.
struct CounterBase {
    const char* ct_arg_name;
    size_t address;
};

// Return the router's four L1 credit counter arrays. The kernel is fed their bases. It sends each pair to the peer
// in one packet, so it relies on the arrays being back to back and the same size; the size comes from that spacing.
manifest::L1CreditCounters collect_credit_counters(const ManifestRouterInputs& inputs) {
    const auto& args = inputs.kernel.named_ct_args;
    // In L1 order
    std::array<CounterBase, 4> bases = {{
        {"TO_SENDER_REMOTE_ACK_COUNTERS_BASE_ADDR"},
        {"TO_SENDER_REMOTE_COMPLETION_COUNTERS_BASE_ADDR"},
        {"LOCAL_RECEIVER_ACK_COUNTERS_BASE_ADDR"},
        {"LOCAL_RECEIVER_COMPLETION_COUNTERS_BASE_ADDR"},
    }};
    for (auto& base : bases) {
        base.address = emitted_value(args, base.ct_arg_name);
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

// A channel's packet slots, each `slot_size` bytes, or none when it has no slots. No struct describes a slot: it
// holds a packet.
std::optional<manifest::content::L1> ring_buffer(
    const ManifestRouterInputs& inputs, size_t address, size_t num_slots, size_t slot_size) {
    if (num_slots == 0) {
        return std::nullopt;
    }
    return manifest::content::L1{
        .address = static_cast<uint32_t>(address),
        .type =
            {
                .element = layout::element::Bytes{},
                .size = static_cast<uint32_t>(num_slots * slot_size),
                .count = static_cast<uint32_t>(num_slots),
            },
        .cleared_by_host = is_cleared_by_host(inputs, address),
    };
}

// The L1 from the end of the channel buffers to the end of all L1.
std::optional<manifest::content::L1> collect_leftover_l1(
    const ManifestRouterInputs& inputs, const FabricStaticSizedChannelsAllocator& allocator) {
    const size_t start = allocator.get_channel_buffers_end_address();
    const size_t end = inputs.erisc_builder.config.max_l1_loading_size;
    TT_FATAL(
        start <= end,
        "Fabric manifest: the channel buffers end at {:#x}, past the end of loadable L1 at {:#x}",
        start,
        end);
    if (start == end) {
        return std::nullopt;
    }
    return manifest::content::L1{
        .address = static_cast<uint32_t>(start),
        .type = {.element = layout::element::Bytes{}, .size = static_cast<uint32_t>(end - start), .count = 0},
        .cleared_by_host = is_cleared_by_host(inputs, start),
    };
}

using manifest::FieldIndex;

// `arg`'s name, with the channel's or edge's index in its {}.
std::string arg_name(const manifest::NamedArg& arg, const FieldIndex& index) {
    switch (arg.index) {
        case manifest::ArgIndex::NONE: return std::string(arg.name);
        case manifest::ArgIndex::COMPACT: return fmt::format(fmt::runtime(arg.name), index.compact);
        case manifest::ArgIndex::FABRIC_POSITION: return fmt::format(fmt::runtime(arg.name), index.fabric_position);
        case manifest::ArgIndex::VC_AND_EDGE: return fmt::format(fmt::runtime(arg.name), index.vc, index.edge);
    }
    TT_THROW("Fabric manifest: unknown argument index {}", static_cast<int>(arg.index));
}

// `field`, with its content's address, stream id or value filled in from `arg`, the value of its source. An L1 address
// of 0 is a buffer the builder did not allocate (FabricRouterDiagnosticBufferMap::BufferRegion::is_enabled).
manifest::Field make_field(const ManifestRouterInputs& inputs, const manifest::RouterField& field, uint32_t arg) {
    const std::string_view key = field.key.value;
    if (std::holds_alternative<manifest::content::L1>(field.content.value) && arg == 0) {
        return {.key = key, .category = field.category.value, .content = std::nullopt};
    }
    manifest::Content content = field.content.value;
    std::visit(
        ttsl::overloaded{
            [&](manifest::content::L1& l1) {
                l1.address = arg;
                l1.cleared_by_host = is_cleared_by_host(inputs, arg);
            },
            [&](manifest::content::Stream& stream) { stream.stream_id = arg; },
            [&](manifest::content::Number& number) { number.value = arg; },
            [&](manifest::content::Flag& flag) {
                TT_FATAL(arg <= 1, "Fabric manifest: field {} is a flag, but is {}", key, arg);
                flag.value = arg != 0;
            },
            [&](manifest::content::Enum& enumerator) {
                TT_FATAL(
                    enumerator.is_enumerator(arg),
                    "Fabric manifest: field {} is {}, which is not a {}",
                    key,
                    arg,
                    manifest::schema_name(enumerator.type));
                enumerator.value = arg;
            },
        },
        content);
    return {
        .key = key,
        .category = field.category.value,
        .content = std::move(content),
    };
}

// What decides whether the builder emits a named argument (Emitted), as the kernel is fed it.
struct Emission {
    bool is_2d_fabric = false;
    bool channel_trimming_capture = false;
};

bool is_emitted(manifest::Emitted emitted, const Emission& emission) {
    switch (emitted) {
        case manifest::Emitted::ALWAYS: return true;
        case manifest::Emitted::FABRIC_2D: return emission.is_2d_fabric;
        case manifest::Emitted::CHANNEL_TRIMMING_CAPTURE: return emission.channel_trimming_capture;
    }
    TT_THROW("Fabric manifest: unknown emission {}", static_cast<int>(emitted));
}

// The fields in `table`, for the channel or edge at `index` when the table is per channel or per edge. `read_arg`
// reads a named argument by its name.
template <typename ReadArg>
std::vector<manifest::Field> collect_fields(
    const ManifestRouterInputs& inputs,
    std::span<const manifest::RouterField> table,
    const FieldIndex& index,
    const Emission& emission,
    const ReadArg& read_arg) {
    std::vector<manifest::Field> fields;
    for (const auto& field : table) {
        const std::optional<uint32_t> value = std::visit(
            ttsl::overloaded{
                [&](const manifest::NamedArg& arg) -> std::optional<uint32_t> {
                    if (!is_emitted(arg.emitted, emission)) {
                        return std::nullopt;
                    }
                    return read_arg(arg_name(arg, index));
                },
                [&](const manifest::BuilderMember& member) -> std::optional<uint32_t> {
                    return member.read(inputs.erisc_builder, index);
                },
            },
            field.source);
        if (value.has_value()) {
            fields.push_back(make_field(inputs, field, *value));
        }
    }
    return fields;
}

// A sender channel's indices. The kernel indexes its per-channel tables by the compact index, over this router's
// own counts, except the free-slots stream table, which it indexes by the fabric position, over the fabric's
// maximum counts (fabric_position_for_compact_sender).
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
    const manifest::RouterShape& shape;
    const FabricStaticSizedChannelsAllocator& allocator;
    const builder::RouterProducerSlots producer_slots;
    const Emission emission;
    const bool speedy_vc0;
    // The router's VC0 senders other than the worker channel are carried by its tensix mux.
    const bool mux_mode;
    // The channel trimming profile the builder applied to this router, if any.
    const std::optional<ChannelTrimmingOverrides>& trimming;
    const uint32_t channel_buffer_size;
};

// The stream register the kernel is fed under `name`.
manifest::content::Stream fed_stream(const ManifestRouterInputs& inputs, const std::string& name) {
    return {
        .stream_id = emitted_value(inputs.kernel.named_ct_args, name),
        .reg = manifest::StreamRegister::BUF_SPACE_AVAILABLE,
        .type = layout::type_of<uint32_t>(),
    };
}

// The kernel runs VC1's channel steps only under FABRIC_2D_VC1_SERVICED, and VC2's only under FABRIC_2D_VC2_SERVICED.
bool kernel_runs_vc(const ManifestRouterInputs& inputs, uint32_t vc) {
    switch (vc) {
        case 1: return inputs.kernel.defines.contains("FABRIC_2D_VC1_SERVICED");
        case 2: return inputs.kernel.defines.contains("FABRIC_2D_VC2_SERVICED");
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
    for (uint32_t risc_id = 0; risc_id < inputs.kernel.named_ct_args.size(); ++risc_id) {
        if (get_named_arg(inputs.kernel.named_ct_args[risc_id], serviced_name) != 0) {
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

// The worker channel is fed by the local worker, or in mux mode by the tensix mux. A sibling producer comes from the
// edge that lands on the channel, which the chip pass sets.
std::optional<manifest::SenderChannelProducer> collect_local_producer(
    const ChannelContext& ctx, const SenderChannelIndex& index) {
    if (ctx.producer_slots.worker_channel(index.vc) != index.channel) {
        return std::nullopt;
    }
    if (ctx.mux_mode && index.vc == 0) {
        return manifest::LocalTensixMux{};
    }
    return manifest::LocalWorker{};
}

// Stream-backed credits are the TO_SENDER_<c>_PKTS_* registers, which the kernel looks up by compact index.
// Counter-backed credits are the channel's element of the to_sender counter arrays. Only VC0 with bubble flow control
// gets first-level acks.
manifest::SenderChannelCredits collect_sender_credits(const ChannelContext& ctx, const SenderChannelIndex& index) {
    const bool counters =
        emitted_flag(ctx.inputs.kernel.named_ct_args, fmt::format("VC{}_USES_COUNTER_CREDITS", index.vc));
    const auto credit = [&](manifest::CreditCounterArray array, const char* stream_name) -> manifest::CreditRef {
        if (counters) {
            return manifest::CounterRef{.array = array, .index = index.compact};
        }
        return fed_stream(ctx.inputs, fmt::format(fmt::runtime(stream_name), index.compact));
    };

    manifest::SenderChannelCredits credits{
        .completed = credit(manifest::CreditCounterArray::TO_SENDER_COMPLETION, "TO_SENDER_{}_PKTS_COMPLETED_ID"),
    };
    if (index.vc == 0 && ctx.shape.vc0_bubble_flow_control) {
        credits.acked = credit(manifest::CreditCounterArray::TO_SENDER_ACK, "TO_SENDER_{}_PKTS_ACKED_ID");
    }
    return credits;
}

manifest::SenderChannel collect_sender_channel(const ChannelContext& ctx, const SenderChannelIndex& index) {
    const auto& inputs = ctx.inputs;
    const auto& args = inputs.kernel.named_ct_args;
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
    sender.producer = collect_local_producer(ctx, index);
    sender.ring_buffer = ring_buffer(
        inputs,
        ctx.allocator.get_sender_channel_base_address(index.vc, index.channel),
        ctx.allocator.get_sender_channel_number_of_slots(index.vc, index.channel),
        ctx.channel_buffer_size);
    sender.credits = collect_sender_credits(ctx, index);
    sender.fields = collect_fields(
        inputs,
        manifest::k_sender_channel_fields,
        {.compact = c, .fabric_position = index.fabric_position},
        ctx.emission,
        [&](const std::string& name) { return emitted_value(args, name); });
    return sender;
}

// The kernel runs VC0's receiver step on channel 0 and VC1's on channel 1. VC2's runs on channel 2 when VC1 has
// senders and on channel 1 otherwise (VC2_RECEIVER_CHANNEL).
uint32_t kernel_receiver_channel(const ChannelContext& ctx, uint32_t vc) {
    if (vc < 2) {
        return vc;
    }
    return ctx.shape.senders_per_vc[1] > 0 ? 2 : 1;
}

// The VC whose downstream edges the kernel gives the receiver's step: its own, except that receiver 0 gets VC1's
// under FABRIC_2D_VC0_CROSSOVER_TO_VC1. The speedy steps, VC0's in speedy mode and VC2's always, are given none.
std::optional<uint32_t> collect_forwards_on(const ChannelContext& ctx, uint32_t vc, bool serviced) {
    if (!serviced || vc == 2 || (vc == 0 && ctx.speedy_vc0)) {
        return std::nullopt;
    }
    if (vc == 0 && ctx.inputs.kernel.defines.contains("FABRIC_2D_VC0_CROSSOVER_TO_VC1")) {
        return 1;
    }
    return vc;
}

manifest::ReceiverChannel collect_receiver_channel(const ChannelContext& ctx, const ReceiverChannelIndex& index) {
    const auto& inputs = ctx.inputs;
    const auto& args = inputs.kernel.named_ct_args;
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
        !serviced || c == kernel_receiver_channel(ctx, index.vc),
        "Fabric manifest: VC{} receiver channel {} is compact channel {}, but the kernel runs VC{}'s receiver on "
        "channel {}",
        index.vc,
        index.channel,
        c,
        index.vc,
        kernel_receiver_channel(ctx, index.vc));
    receiver.forwards_on = collect_forwards_on(ctx, index.vc, serviced);
    receiver.ring_buffer = ring_buffer(
        inputs,
        ctx.allocator.get_receiver_channel_base_address(index.vc, index.channel),
        ctx.allocator.get_receiver_channel_number_of_slots(index.vc, index.channel),
        ctx.channel_buffer_size);
    receiver.fields = collect_fields(
        inputs, manifest::k_receiver_channel_fields, {.compact = c}, ctx.emission, [&](const std::string& name) {
            return emitted_value(args, name);
        });
    return receiver;
}

ChannelContext make_channel_context(const ManifestRouterInputs& inputs, const manifest::RouterShape& shape) {
    const auto& args = inputs.kernel.named_ct_args;
    const auto* allocator =
        dynamic_cast<const FabricStaticSizedChannelsAllocator*>(inputs.erisc_builder.config.channel_allocator.get());
    TT_FATAL(allocator != nullptr, "Fabric manifest: the router's channel allocator is not statically sized");
    return {
        .inputs = inputs,
        .shape = shape,
        .allocator = *allocator,
        .producer_slots = builder::RouterProducerSlots(
            builder::routing_direction_to_eth_direction(inputs.location.direction), shape.senders_per_vc),
        .emission =
            {
                .is_2d_fabric = emitted_flag(args, "IS_2D_FABRIC"),
                .channel_trimming_capture = emitted_flag(args, "ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE"),
            },
        .speedy_vc0 = emitted_flag(args, "ENABLE_SPEEDY_VC0"),
        .mux_mode = emitted_flag(args, "FABRIC_TENSIX_EXTENSION_MUX_MODE"),
        .trimming = inputs.erisc_builder.get_channel_trimming_overrides(),
        .channel_buffer_size = emitted_value(args, "CHANNEL_BUFFER_SIZE"),
    };
}

// Where VC `vc`'s senders start in the fabric's free-slots stream table: VC0's at 0, the others where the kernel is
// told (fabric_position_for_compact_sender).
uint32_t fabric_position_start(const ManifestRouterInputs& inputs, uint32_t vc) {
    if (vc == 0) {
        return 0;
    }
    return emitted_value(inputs.kernel.named_ct_args, fmt::format("VC{}_FABRIC_POSITION_START", vc));
}

// Every sender and receiver channel in the router's shape, each indexed [vc][channel]. Compact indices run over the
// shape's counts, VC by VC.
manifest::Channels collect_channels(const ChannelContext& ctx) {
    const auto& shape = ctx.shape;
    manifest::Channels channels;
    channels.senders.resize(builder_config::MAX_NUM_VCS);
    channels.receivers.resize(builder_config::MAX_NUM_VCS);
    uint32_t sender_compact = 0;
    uint32_t receiver_compact = 0;
    for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        const uint32_t position_start = fabric_position_start(ctx.inputs, vc);
        for (uint32_t channel = 0; channel < shape.senders_per_vc[vc]; ++channel) {
            const SenderChannelIndex index{
                .vc = vc,
                .channel = channel,
                .compact = sender_compact++,
                .fabric_position = position_start + channel,
            };
            channels.senders[vc].push_back(collect_sender_channel(ctx, index));
        }
        for (uint32_t channel = 0; channel < shape.receivers_per_vc[vc]; ++channel) {
            const ReceiverChannelIndex index{
                .vc = vc,
                .channel = channel,
                .compact = receiver_compact++,
            };
            channels.receivers[vc].push_back(collect_receiver_channel(ctx, index));
        }
    }
    return channels;
}

// The router's edges on `vc`, ordered by edge. The kernel is fed how many there are (NUM_DOWNSTREAM_SENDERS_VC<vc>)
// and which slots they are in, and the adapter names their targets. The kernel takes an edge's teardown semaphore by
// its rank among the router's edges, VC0's before VC1's. It builds VC1's edges only under FABRIC_2D_VC1_ACTIVE, and
// gives VC2 none.
std::vector<manifest::DownstreamEdge> collect_downstream_edges(const ChannelContext& ctx, uint32_t vc) {
    // The kernel never forwards on VC2.
    if (vc == 2) {
        return {};
    }
    const auto& inputs = ctx.inputs;
    const auto& args = inputs.kernel.named_ct_args;
    const auto& adapter = *inputs.erisc_builder.receiver_channel_to_downstream_adapter;
    const auto& connections = adapter.get_downstream_connections(vc);
    const uint32_t num_edges = emitted_value(args, fmt::format("NUM_DOWNSTREAM_SENDERS_VC{}", vc));
    TT_FATAL(
        connections.size() == num_edges,
        "Fabric manifest: the kernel is fed {} VC{} downstream edges, but the adapter names {} targets",
        num_edges,
        vc,
        connections.size());
    TT_FATAL(
        vc == 0 || num_edges == 0 || inputs.kernel.defines.contains("FABRIC_2D_VC1_ACTIVE"),
        "Fabric manifest: the router has VC1 downstream edges, but no FABRIC_2D_VC1_ACTIVE to build them");

    const auto my_direction = builder::routing_direction_to_eth_direction(inputs.location.direction);
    std::vector<manifest::DownstreamEdge> edges;
    uint32_t mask = 0;
    for (const auto& [direction, core] : connections) {
        const uint32_t slot =
            ctx.emission.is_2d_fabric ? get_receiver_channel_compact_index(my_direction, direction) : 0;
        TT_FATAL((mask & (1u << slot)) == 0, "Fabric manifest: VC{} has two downstream edges in slot {}", vc, slot);
        mask |= 1u << slot;
        edges.push_back({
            .edge = slot + 1,
            .target = {.direction = direction},
            .landing_channel =
                builder::get_downstream_sender_channel_for_vc(ctx.emission.is_2d_fabric, vc, my_direction, direction),
            .core = core,
        });
    }
    // Otherwise the manifest would number the edges differently from the kernel.
    TT_FATAL(
        mask == adapter.get_downstream_edm_mask_for_vc(vc),
        "Fabric manifest: VC{}'s downstream edges are in slots {:#x}, but the kernel is given {:#x}",
        vc,
        mask,
        adapter.get_downstream_edm_mask_for_vc(vc));
    std::ranges::sort(edges, {}, &manifest::DownstreamEdge::edge);

    const uint32_t teardown_base = vc == 1 ? emitted_value(args, "NUM_DOWNSTREAM_SENDERS_VC0") : 0;
    for (uint32_t rank = 0; rank < edges.size(); ++rank) {
        auto& edge = edges[rank];
        edge.fields = collect_fields(
            inputs,
            manifest::k_downstream_edge_fields,
            {.vc = vc, .edge = edge.edge, .teardown = teardown_base + rank},
            ctx.emission,
            [&](const std::string& name) { return emitted_value(args, name); });
    }
    return edges;
}

std::vector<std::vector<manifest::DownstreamEdge>> collect_router_edges(const ChannelContext& ctx) {
    std::vector<std::vector<manifest::DownstreamEdge>> edges;
    for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        edges.push_back(collect_downstream_edges(ctx, vc));
    }
    return edges;
}

}  // namespace

manifest::Router collect_manifest_router(const ManifestRouterInputs& inputs) {
    const auto& args = inputs.kernel.named_ct_args;
    TT_FATAL(
        !args.empty() && inputs.kernel.processors.size() == args.size(),
        "Fabric manifest: got compile-time arguments for {} RISCs and processors for {}",
        args.size(),
        inputs.kernel.processors.size());
    auto shape = collect_shape(inputs);
    TT_FATAL(
        shape.num_active_eriscs == args.size(),
        "Fabric manifest: the kernel runs on {} ERISCs, but got compile-time arguments for {}",
        shape.num_active_eriscs,
        args.size());

    const auto ctx = make_channel_context(inputs, shape);
    auto channels = collect_channels(ctx);
    auto edges = collect_router_edges(ctx);

    auto fields = collect_fields(inputs, manifest::k_router_fields, {}, ctx.emission, [&](const std::string& name) {
        return emitted_value(args, name);
    });
    std::vector<manifest::Erisc> eriscs;
    for (size_t risc_id = 0; risc_id < args.size(); ++risc_id) {
        eriscs.push_back({
            .processor = inputs.kernel.processors[risc_id],
            .fields = collect_fields(
                inputs,
                manifest::k_erisc_fields,
                {},
                ctx.emission,
                [&](const std::string& name) { return get_named_arg(args[risc_id], name); }),
        });
    }
    return {
        .identity = collect_identity(inputs.location),
        .link = collect_link(inputs),
        .shape = shape,
        .credit_counters = collect_credit_counters(inputs),
        .channels = std::move(channels),
        .intra_chip_downstream_edges = std::move(edges),
        .fields = std::move(fields),
        .eriscs = std::move(eriscs),
        .leftover_l1 = collect_leftover_l1(inputs, ctx.allocator),
    };
}

}  // namespace tt::tt_fabric
