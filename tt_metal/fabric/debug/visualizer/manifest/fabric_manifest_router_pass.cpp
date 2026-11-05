// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_router_pass.hpp"

#include <fmt/format.h>
#include <hostdevcommon/fabric_common.h>
#include <tt_stl/assert.hpp>
#include <tt_stl/overloaded.hpp>
#include <algorithm>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>

#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp"
#include "tt_metal/fabric/builder/protected_domain_effect.hpp"
#include "tt_metal/fabric/builder/router_wiring_rules.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/fabric_router_builder.hpp"
#include "tt_metal/fabric/packet_header_types.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_fields.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"
#include "tt_metal/llrt/hal.hpp"

namespace tt::tt_fabric {

namespace {

using manifest::RouterArgs;

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

// Return the RouterShape. The kernel is given its ERISC count, its sender channels per VC and whether it has a tensix
// extension, but only the total of its receivers, so the receivers per VC and the number of VCs come from the VC shape.
manifest::RouterShape collect_shape(const ManifestRouterInputs& inputs) {
    const RouterArgs& args = inputs.args;
    manifest::RouterShape shape{
        .num_vcs = inputs.vc_shape.num_vcs,
        .receivers_per_vc = inputs.vc_shape.receiver_counts,
        .num_active_eriscs = args.read_router_arg("NUM_ACTIVE_ERISCS"),
        .channel_trimming_overrides_applied = inputs.erisc_builder.get_channel_trimming_overrides().has_value(),
        .has_tensix_extension = args.read_router_arg("FABRIC_TENSIX_EXTENSION_MUX_MODE") != 0,
    };
    for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        shape.senders_per_vc[vc] = args.read_router_arg(fmt::format("ACTUAL_VC{}_SENDER_CHANNELS", vc));
    }
    return shape;
}

// An array of T filling `size` bytes at `address`.
template <typename T>
manifest::content::L1 describe_l1_array(size_t address, size_t size) {
    TT_FATAL(
        size % sizeof(T) == 0,
        "Fabric manifest: array at {:#x} is {} bytes, not a whole number of {}-byte elements",
        address,
        size,
        sizeof(T));
    return manifest::register_l1_region<T>(static_cast<uint32_t>(address), static_cast<uint32_t>(size / sizeof(T)));
}

// Return the router's L1 credit counter arrays, from the bases the kernel is given.
manifest::L1CreditCounters collect_credit_counters(const ManifestRouterInputs& inputs) {
    const auto read_base = [&](const char* name) -> size_t { return inputs.args.read_router_arg(name); };
    const size_t to_sender_ack = read_base("TO_SENDER_REMOTE_ACK_COUNTERS_BASE_ADDR");
    const size_t to_sender_completion = read_base("TO_SENDER_REMOTE_COMPLETION_COUNTERS_BASE_ADDR");
    const size_t receiver_ack = read_base("LOCAL_RECEIVER_ACK_COUNTERS_BASE_ADDR");
    const size_t receiver_completion = read_base("LOCAL_RECEIVER_COMPLETION_COUNTERS_BASE_ADDR");
    const size_t size = to_sender_completion - to_sender_ack;
    return {
        .to_sender_ack = describe_l1_array<uint32_t>(to_sender_ack, size),
        .to_sender_completion = describe_l1_array<uint32_t>(to_sender_completion, size),
        .receiver_ack = describe_l1_array<uint32_t>(receiver_ack, size),
        .receiver_completion = describe_l1_array<uint32_t>(receiver_completion, size),
    };
}

// A channel's packet slots, each `slot_size` bytes, or none when it has no slots. Each slot starts with the run's
// packet header, and the payload after it is left as raw bytes.
std::optional<manifest::content::L1> describe_ring_buffer(
    const PacketHeaderConfig& packet_header_config, size_t address, size_t num_slots, size_t slot_size) {
    if (num_slots == 0) {
        return std::nullopt;
    }
    return visit_packet_header_type(packet_header_config, [&]<typename Header>(std::type_identity<Header>) {
        return manifest::register_l1_region<Header>(
            static_cast<uint32_t>(address), static_cast<uint32_t>(num_slots), static_cast<uint32_t>(slot_size));
    });
}

// The L1 from the end of what the router places to max_l1_loading_size, the end of the L1 the router may load into.
std::optional<manifest::content::L1> collect_leftover_l1(const ManifestRouterInputs& inputs) {
    const size_t start = inputs.erisc_builder.get_allocated_l1_end_address();
    const size_t end = inputs.erisc_builder.config.max_l1_loading_size;
    if (start == end) {
        return std::nullopt;
    }
    return manifest::register_l1_region<std::byte>(static_cast<uint32_t>(start), static_cast<uint32_t>(end - start), 1);
}

// The described stream register `stream_reg` on stream `stream_id`: where it is, from the HAL, and how many of its low
// bits hold its value.
manifest::content::StreamReg describe_stream_register(
    const tt::tt_metal::Hal& hal, uint32_t stream_id, const manifest::StreamRegField& stream_reg) {
    TT_FATAL(
        hal.has_stream_registers(),
        "Fabric manifest: the HAL gives no stream register addresses on {}",
        manifest::lower_enum_name(hal.get_arch()));
    TT_FATAL(
        stream_id < StreamRegAssignments::num_eth_stream_registers,
        "Fabric manifest: stream id {} is not one of the {} stream registers",
        stream_id,
        StreamRegAssignments::num_eth_stream_registers);
    // The register's index within its stream's block, and how many of its low bits hold its value.
    const auto [reg_index, value_bits] = [&]() -> std::pair<uint32_t, uint32_t> {
        switch (stream_reg.reg) {
            case manifest::StreamRegister::BUF_SPACE_AVAILABLE:
                return {
                    hal.get_noc_stream_remote_dest_buf_space_available_reg_index(),
                    hal.get_noc_stream_remote_dest_words_free_width()};
            case manifest::StreamRegister::REMOTE_SRC:
                // The fabric's scratch register: four 6-bit fields on Wormhole and Blackhole (fabric_stream_regs.hpp).
                return {hal.get_noc_stream_remote_src_reg_index(), 24};
        }
        TT_THROW("Fabric manifest: unknown stream register {}", static_cast<int>(stream_reg.reg));
    }();
    return {
        .stream_id = stream_id,
        .address = hal.get_noc_overlay_start_addr() + (stream_id * hal.get_noc_stream_reg_space_size()) +
                   static_cast<uint32_t>(reg_index * sizeof(uint32_t)),
        .value_bits = value_bits,
        .type = stream_reg.type,
    };
}

using manifest::FieldIndex;

// `arg`'s name, with the channel's or edge's index in its {}.
std::string get_arg_name(const manifest::NamedArg& arg, const FieldIndex& index) {
    switch (arg.index) {
        case manifest::ArgIndex::NONE: return std::string(arg.name);
        case manifest::ArgIndex::COMPACT: return fmt::format(fmt::runtime(arg.name), index.compact);
        case manifest::ArgIndex::FABRIC_POSITION: return fmt::format(fmt::runtime(arg.name), index.fabric_position);
        case manifest::ArgIndex::VC_AND_EDGE: return fmt::format(fmt::runtime(arg.name), index.vc, index.edge);
    }
    TT_THROW("Fabric manifest: unknown argument index {}", static_cast<int>(arg.index));
}

// `field`, with its content's address, stream register or value filled in from `arg`, the value of its source. An L1
// address of 0 is a buffer the builder did not allocate (FabricRouterDiagnosticBufferMap::BufferRegion::is_enabled),
// and k_unused_stream_id a stream register it did not allocate.
manifest::Field make_field(const ManifestRouterInputs& inputs, const manifest::RouterField& field, uint32_t arg) {
    const std::string_view key = field.key.value;
    const bool unallocated =
        (std::holds_alternative<manifest::content::L1>(field.content.value) && arg == 0) ||
        (std::holds_alternative<manifest::StreamRegField>(field.content.value) && arg == k_unused_stream_id);
    if (unallocated) {
        return {.key = key, .category = field.category.value, .content = std::nullopt};
    }
    manifest::Content content = std::visit(
        ttsl::overloaded{
            [&](manifest::content::L1 l1) -> manifest::Content {
                l1.address = arg;
                return l1;
            },
            [&](const manifest::StreamRegField& stream) -> manifest::Content {
                return describe_stream_register(inputs.hal, arg, stream);
            },
            [&](manifest::content::Number number) -> manifest::Content {
                number.value = arg;
                return number;
            },
            [&](manifest::content::Flag flag) -> manifest::Content {
                TT_FATAL(arg <= 1, "Fabric manifest: field {} is a flag, but is {}", key, arg);
                flag.value = arg != 0;
                return flag;
            },
            [&](manifest::content::Enum enumerator) -> manifest::Content {
                const std::optional<std::string_view> value_name = enumerator.name_value(arg);
                TT_FATAL(
                    value_name.has_value(),
                    "Fabric manifest: field {} is {}, which is none of {}'s enumerators",
                    key,
                    arg,
                    enumerator.type);
                enumerator.value = arg;
                enumerator.value_name = *value_name;
                return enumerator;
            },
        },
        field.content.value);
    return {
        .key = key,
        .category = field.category.value,
        .content = content,
    };
}

// What decides whether the builder emits a named argument (Emitted), as one ERISC is fed it.
struct Emission {
    bool is_2d_fabric = false;
    bool channel_trimming_capture = false;
};

Emission read_emission(const RouterArgs& args, uint32_t erisc) {
    return {
        .is_2d_fabric = args.read_erisc_arg(erisc, "IS_2D_FABRIC") != 0,
        .channel_trimming_capture = args.read_erisc_arg(erisc, "ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE") != 0,
    };
}

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
                    return read_arg(get_arg_name(arg, index));
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

// `table`'s fields as each ERISC is fed them, for the channel or edge at `index`.
manifest::FieldsByErisc collect_fields_by_erisc(
    const ManifestRouterInputs& inputs, std::span<const manifest::RouterField> table, const FieldIndex& index) {
    const RouterArgs& args = inputs.args;
    manifest::FieldsByErisc fields_by_erisc;
    fields_by_erisc.reserve(args.num_eriscs());
    for (uint32_t erisc = 0; erisc < args.num_eriscs(); ++erisc) {
        fields_by_erisc.push_back(
            collect_fields(inputs, table, index, read_emission(args, erisc), [&](const std::string& name) {
                return args.read_erisc_arg(erisc, name);
            }));
    }
    return fields_by_erisc;
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
    const bool is_2d_fabric;
    // The channel trimming profile the builder applied to this router, if any.
    const std::optional<ChannelTrimmingOverrides>& trimming;
    const uint32_t channel_buffer_size;
};

// The stream register stream `stream_id`'s credits come back on, or null when the builder gave the credit the unused
// stream id.
std::optional<manifest::content::StreamReg> describe_credit_stream(
    const ManifestRouterInputs& inputs, uint32_t stream_id) {
    if (stream_id == k_unused_stream_id) {
        return std::nullopt;
    }
    return describe_stream_register(
        inputs.hal,
        stream_id,
        manifest::register_stream_region<uint32_t>(manifest::StreamRegister::BUF_SPACE_AVAILABLE));
}

// The kernel runs VC1's channel steps only under FABRIC_2D_VC1_SERVICED, and VC2's only under FABRIC_2D_VC2_SERVICED.
bool is_vc_run_by_kernel(const ManifestRouterInputs& inputs, uint32_t vc) {
    switch (vc) {
        case 1: return inputs.args.has_define("FABRIC_2D_VC1_SERVICED");
        case 2: return inputs.args.has_define("FABRIC_2D_VC2_SERVICED");
        default: return true;
    }
}

// The ERISCs that run a channel: those it was assigned to, or none when the kernel does not run the VC's steps.
std::vector<uint32_t> collect_serviced_by(
    const ManifestRouterInputs& inputs, uint32_t vc, const std::vector<uint32_t>& assigned_to) {
    return is_vc_run_by_kernel(inputs, vc) ? assigned_to : std::vector<uint32_t>{};
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
    if (!is_vc_run_by_kernel(ctx.inputs, vc)) {
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
// edge that lands on the channel, which the topology pass sets.
std::optional<manifest::SenderChannelProducer> collect_local_producer(
    const ChannelContext& ctx, const SenderChannelIndex& index) {
    if (ctx.producer_slots.worker_channel(index.vc) != index.channel) {
        return std::nullopt;
    }
    if (ctx.shape.has_tensix_extension && index.vc == 0) {
        return manifest::LocalTensixMux{};
    }
    return manifest::LocalWorker{};
}

// A sender channel's credits as each ERISC is told to read them. A VC whose credits use L1 counters
// (VC<v>_USES_COUNTER_CREDITS) reads the channel's element of the to_sender counter arrays; otherwise the credits come
// back on the TO_SENDER_<c>_PKTS_* stream registers, by compact index. First-level acks come back only where the ERISC
// is fed ENABLE_FIRST_LEVEL_ACK_VC<v> set. The builder feeds it for VC0 and VC1, and clears VC0's wherever it picks the
// speedy step, which has no acks; VC2's step is always the speedy one, so VC2 gets no flag.
std::vector<manifest::SenderChannelCredits> collect_sender_credits(
    const ChannelContext& ctx, const SenderChannelIndex& index) {
    const RouterArgs& args = ctx.inputs.args;
    const bool vc_uses_counters = args.read_router_arg(fmt::format("VC{}_USES_COUNTER_CREDITS", index.vc)) != 0;
    const auto credit = [&](manifest::CreditCounterArray array, const char* stream_name) -> manifest::CreditRef {
        if (vc_uses_counters) {
            return manifest::CounterRef{.array = array, .index = index.compact};
        }
        return describe_credit_stream(
            ctx.inputs, args.read_router_arg(fmt::format(fmt::runtime(stream_name), index.compact)));
    };

    const std::string first_level_ack = fmt::format("ENABLE_FIRST_LEVEL_ACK_VC{}", index.vc);

    std::vector<manifest::SenderChannelCredits> credits_by_erisc;
    credits_by_erisc.reserve(args.num_eriscs());
    for (uint32_t erisc = 0; erisc < args.num_eriscs(); ++erisc) {
        manifest::SenderChannelCredits credits{
            .completed = credit(manifest::CreditCounterArray::TO_SENDER_COMPLETION, "TO_SENDER_{}_PKTS_COMPLETED_ID"),
        };
        if (args.find_erisc_arg(erisc, first_level_ack).value_or(0) != 0) {
            credits.acked = credit(manifest::CreditCounterArray::TO_SENDER_ACK, "TO_SENDER_{}_PKTS_ACKED_ID");
        }
        credits_by_erisc.push_back(credits);
    }
    return credits_by_erisc;
}

manifest::SenderChannel collect_sender_channel(const ChannelContext& ctx, const SenderChannelIndex& index) {
    const auto& inputs = ctx.inputs;
    const uint32_t c = index.compact;

    manifest::SenderChannel sender;
    sender.serviced_by = collect_serviced_by(inputs, index.vc, inputs.args.find_eriscs_servicing_sender(c));
    sender.status = collect_status(
        ctx,
        "sender",
        index.vc,
        index.channel,
        sender.serviced_by,
        ctx.shape.has_tensix_extension && index.vc == 0 && ctx.producer_slots.worker_channel(0) != index.channel,
        ctx.trimming.has_value() && !ctx.trimming->is_sender_channel_used(c));
    sender.producer = collect_local_producer(ctx, index);
    sender.ring_buffer = describe_ring_buffer(
        inputs.packet_header,
        ctx.allocator.get_sender_channel_base_address(index.vc, index.channel),
        ctx.allocator.get_sender_channel_number_of_slots(index.vc, index.channel),
        ctx.channel_buffer_size);
    sender.credits_by_erisc = collect_sender_credits(ctx, index);
    sender.fields_by_erisc = collect_fields_by_erisc(
        inputs, manifest::k_sender_channel_fields, {.compact = c, .fabric_position = index.fabric_position});
    return sender;
}

// The VC whose downstream edges each ERISC gives the receiver's step: its own, except that receiver 0 gets VC1's under
// FABRIC_2D_VC0_CROSSOVER_TO_VC1. Null on an ERISC that doesn't run the step, and for the speedy steps, which are given
// none: VC2's always, and VC0's on an ERISC in speedy mode (its ENABLE_SPEEDY_VC0).
std::vector<std::optional<uint32_t>> collect_forwards_on(
    const ChannelContext& ctx, uint32_t vc, const std::vector<uint32_t>& serviced_by) {
    const RouterArgs& args = ctx.inputs.args;
    std::vector<std::optional<uint32_t>> forwards_on_by_erisc;
    forwards_on_by_erisc.reserve(args.num_eriscs());
    for (uint32_t erisc = 0; erisc < args.num_eriscs(); ++erisc) {
        const bool runs_step = std::ranges::find(serviced_by, erisc) != serviced_by.end();
        const bool speedy_vc0 = args.read_erisc_arg(erisc, "ENABLE_SPEEDY_VC0") != 0;
        if (!runs_step || vc == 2 || (vc == 0 && speedy_vc0)) {
            forwards_on_by_erisc.emplace_back(std::nullopt);
        } else if (vc == 0 && args.has_define("FABRIC_2D_VC0_CROSSOVER_TO_VC1")) {
            forwards_on_by_erisc.emplace_back(1);
        } else {
            forwards_on_by_erisc.emplace_back(vc);
        }
    }
    return forwards_on_by_erisc;
}

manifest::ReceiverChannel collect_receiver_channel(const ChannelContext& ctx, const ReceiverChannelIndex& index) {
    const auto& inputs = ctx.inputs;
    const uint32_t c = index.compact;

    manifest::ReceiverChannel receiver;
    receiver.serviced_by = collect_serviced_by(inputs, index.vc, inputs.args.find_eriscs_servicing_receiver(c));
    receiver.status = collect_status(
        ctx,
        "receiver",
        index.vc,
        index.channel,
        receiver.serviced_by,
        /*carried_by_mux=*/false,
        ctx.trimming.has_value() && !ctx.trimming->is_receiver_channel_data_forwarded(c));
    receiver.forwards_on_by_erisc = collect_forwards_on(ctx, index.vc, receiver.serviced_by);
    receiver.ring_buffer = describe_ring_buffer(
        inputs.packet_header,
        ctx.allocator.get_receiver_channel_base_address(index.vc, index.channel),
        ctx.allocator.get_receiver_channel_number_of_slots(index.vc, index.channel),
        ctx.channel_buffer_size);
    receiver.fields_by_erisc = collect_fields_by_erisc(inputs, manifest::k_receiver_channel_fields, {.compact = c});
    return receiver;
}

ChannelContext make_channel_context(const ManifestRouterInputs& inputs, const manifest::RouterShape& shape) {
    const auto& builder = inputs.erisc_builder;
    const auto* allocator =
        dynamic_cast<const FabricStaticSizedChannelsAllocator*>(inputs.erisc_builder.config.channel_allocator.get());
    TT_FATAL(allocator != nullptr, "Fabric manifest: the router's channel allocator is not statically sized");
    return {
        .inputs = inputs,
        .shape = shape,
        .allocator = *allocator,
        .producer_slots = builder::RouterProducerSlots(
            builder::routing_direction_to_eth_direction(inputs.location.direction), shape.senders_per_vc),
        .is_2d_fabric = inputs.args.read_router_arg("IS_2D_FABRIC") != 0,
        .trimming = builder.get_channel_trimming_overrides(),
        .channel_buffer_size = inputs.args.read_router_arg("CHANNEL_BUFFER_SIZE"),
    };
}

// Where VC `vc`'s senders start in the fabric's free-slots stream table: VC0's at 0, the others where the kernel is
// told (fabric_position_for_compact_sender).
uint32_t get_fabric_position_start(const ChannelContext& ctx, uint32_t vc) {
    if (vc == 0) {
        return 0;
    }
    return ctx.inputs.args.read_router_arg(fmt::format("VC{}_FABRIC_POSITION_START", vc));
}

// Every sender and receiver channel in the router's shape, each indexed [vc][channel]. Each channel's compact index is
// the flat id the builder numbers it by (RouterVcShape::flat_sender_id, flat_receiver_id).
manifest::RouterChannels collect_channels(const ChannelContext& ctx) {
    const auto& shape = ctx.shape;
    manifest::RouterChannels channels;
    channels.senders.resize(builder_config::MAX_NUM_VCS);
    channels.receivers.resize(builder_config::MAX_NUM_VCS);
    const auto& vc_shape = ctx.inputs.vc_shape;
    for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        const uint32_t position_start = get_fabric_position_start(ctx, vc);
        for (uint32_t channel = 0; channel < shape.senders_per_vc[vc]; ++channel) {
            const SenderChannelIndex index{
                .vc = vc,
                .channel = channel,
                .compact = vc_shape.flat_sender_id(vc, channel),
                .fabric_position = position_start + channel,
            };
            channels.senders[vc].push_back(collect_sender_channel(ctx, index));
        }
        for (uint32_t channel = 0; channel < shape.receivers_per_vc[vc]; ++channel) {
            const ReceiverChannelIndex index{
                .vc = vc,
                .channel = channel,
                .compact = vc_shape.flat_receiver_id(vc, channel),
            };
            channels.receivers[vc].push_back(collect_receiver_channel(ctx, index));
        }
    }
    return channels;
}

// The router's edges on `vc`, ordered by edge.
std::vector<manifest::DownstreamEdge> collect_downstream_edges(const ChannelContext& ctx, uint32_t vc) {
    const auto& inputs = ctx.inputs;
    const auto my_direction = builder::routing_direction_to_eth_direction(inputs.location.direction);
    std::vector<manifest::DownstreamEdge> edges;
    uint32_t mask = 0;
    for (const auto& connection : inputs.connections) {
        if (connection.source_vc != vc) {
            continue;
        }
        const auto direction = builder::routing_direction_to_eth_direction(connection.dest_direction);
        // The adapter numbers the edge by the same rule (add_downstream_connection).
        const uint32_t slot = ctx.is_2d_fabric ? get_receiver_channel_compact_index(my_direction, direction) : 0;
        TT_FATAL((mask & (1u << slot)) == 0, "Fabric manifest: VC{} has two downstream edges in slot {}", vc, slot);
        mask |= 1u << slot;
        edges.push_back({
            .edge = slot + 1,
            .target = {.direction = direction},
            .target_eth_chan = connection.dest_eth_chan,
            .landing_channel = connection.dest_vc_sender_channel,
            .through_tensix_mux = connection.dest_is_tensix,
        });
    }
    std::ranges::sort(edges, {}, &manifest::DownstreamEdge::edge);

    // VC1's teardown semaphores follow VC0's, one per VC0 edge.
    const uint32_t teardown_base = vc == 1 ? inputs.args.read_router_arg("NUM_DOWNSTREAM_SENDERS_VC0") : 0;
    for (uint32_t rank = 0; rank < edges.size(); ++rank) {
        auto& edge = edges[rank];
        edge.fields_by_erisc = collect_fields_by_erisc(
            inputs,
            manifest::k_downstream_edge_fields,
            {.vc = vc, .edge = edge.edge, .teardown = teardown_base + rank});
    }
    return edges;
}

std::vector<std::vector<manifest::DownstreamEdge>> collect_router_edges(const ChannelContext& ctx) {
    std::vector<std::vector<manifest::DownstreamEdge>> edges;
    edges.reserve(builder_config::MAX_NUM_VCS);
    for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        edges.push_back(collect_downstream_edges(ctx, vc));
    }
    return edges;
}

}  // namespace

void check_kernel_inputs_accounted(std::span<const KernelInputs> kernel_inputs_by_erisc) {
    for (const KernelInputs& erisc_inputs : kernel_inputs_by_erisc) {
        for (const auto& [name, value] : erisc_inputs.named_compile_args) {
            TT_FATAL(
                manifest::manifest_reads_arg(name) || manifest::is_unrecorded_arg(name),
                "Fabric manifest: {} is fed to the router, but neither a field table nor the router pass reads it; add "
                "a table entry, or add it to k_unrecorded_args with a reason (fabric_manifest_fields.hpp)",
                name);
        }
        for (const auto& [name, value] : erisc_inputs.defines) {
            TT_FATAL(
                manifest::manifest_reads_define(name) || manifest::is_unrecorded_define(name),
                "Fabric manifest: the define {} is fed to the router, but the router pass does not read it; read it "
                "and add it to k_router_pass_defines, or add it to k_unrecorded_defines with a reason "
                "(fabric_manifest_fields.hpp)",
                name);
        }
    }
}

manifest::Router collect_manifest_router(const ManifestRouterInputs& inputs) {
    const RouterArgs& args = inputs.args;
    check_kernel_inputs_accounted(args.get_kernel_inputs_by_erisc());
    auto shape = collect_shape(inputs);
    const auto ctx = make_channel_context(inputs, shape);
    auto channels = collect_channels(ctx);
    auto edges = collect_router_edges(ctx);

    auto fields_by_erisc = collect_fields_by_erisc(inputs, manifest::k_router_fields, {});
    std::vector<manifest::Erisc> eriscs;
    eriscs.reserve(args.num_eriscs());
    for (const KernelInputs& erisc_inputs : args.get_kernel_inputs_by_erisc()) {
        const auto processor = erisc_inputs.processor;
        eriscs.push_back({
            .processor = processor,
            .physical_erisc = inputs.hal.get_active_eth_physical_erisc(static_cast<uint32_t>(processor)),
        });
    }
    return {
        .identity =
            {
                .eth_chan = inputs.location.eth_chan,
                .direction = builder::routing_direction_to_eth_direction(inputs.location.direction),
            },
        .link = collect_link(inputs),
        .shape = shape,
        .credit_counters = collect_credit_counters(inputs),
        .channels = std::move(channels),
        .intra_chip_downstream_edges = std::move(edges),
        .fields_by_erisc = std::move(fields_by_erisc),
        .eriscs = std::move(eriscs),
        .leftover_l1 = collect_leftover_l1(inputs),
    };
}

}  // namespace tt::tt_fabric
