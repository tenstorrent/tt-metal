// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/fabric_manifest_router_instance.hpp"

#include <algorithm>
#include <limits>
#include <string_view>
#include <tuple>
#include <utility>

#include <fmt/format.h>
#include <tt-metalium/hal.hpp>
#include <tt_stl/assert.hpp>

#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/fabric_router_builder.hpp"
#include "tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h"

namespace tt::tt_fabric {

std::string direction_to_str(eth_chan_directions direction) {
    switch (direction) {
        case eth_chan_directions::EAST: return "E";
        case eth_chan_directions::WEST: return "W";
        case eth_chan_directions::NORTH: return "N";
        case eth_chan_directions::SOUTH: return "S";
        case eth_chan_directions::Z: return "Z";
        default: TT_FATAL(false, "Cannot name sender producer direction {}", static_cast<int>(direction));
    }
}

namespace {

using NamedArgs = std::unordered_map<std::string, uint32_t>;

uint32_t narrow_to_u32(size_t value, std::string_view name) {
    TT_FATAL(
        value <= std::numeric_limits<uint32_t>::max(),
        "{} value {} does not fit in the debug manifest ABI",
        name,
        value);
    return static_cast<uint32_t>(value);
}

uint32_t get_named_arg(const NamedArgs& args, const std::string& name) {
    const auto it = args.find(name);
    TT_FATAL(it != args.end(), "Missing fabric router named compile-time argument {}", name);
    return it->second;
}

void require_named_address(const NamedArgs& args, const std::string& name, size_t expected) {
    TT_FATAL(
        get_named_arg(args, name) == expected,
        "Fabric router debug layout address mismatch: named argument {} is {}, config says {}",
        name,
        get_named_arg(args, name),
        expected);
}

std::vector<ManifestDownstreamEdge> collect_manifest_downstream_edges(
    const FabricEriscDatamoverBuilder& builder, uint32_t vc) {
    return builder.get_manifest_downstream_edges(vc);
}

// Per flat sender channel, the producer intent ("worker" for the local-worker slot, a direction
// name for producer slots, nullopt where unmapped), resolved from the canonical slot rule so
// manifest consumers never re-derive it. Iterates vc/channel in flat order.
std::vector<std::optional<std::string>> collect_sender_producer_names(
    eth_chan_directions facing,
    const std::array<uint32_t, builder_config::MAX_NUM_VCS>& sender_counts,
    const FabricStaticSizedChannelsAllocator* allocator) {
    builder::RouterProducerSlots slots(facing, sender_counts);
    std::vector<std::optional<std::string>> names;
    for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        for (uint32_t channel = 0; channel < allocator->get_num_sender_channels(vc); ++channel) {
            if (slots.worker_channel(vc) == channel) {
                names.emplace_back("worker");
            } else if (const auto producer = slots.producer_at(vc, channel); producer.has_value()) {
                names.emplace_back(direction_to_str(*producer));
            } else {
                names.emplace_back(std::nullopt);
            }
        }
    }
    return names;
}

ManifestRouterRegion make_group_region(std::string id, std::string parent = {}) {
    return {
        .id = std::move(id),
        .parent = std::move(parent),
        .backing = ManifestRegionBacking::GROUP,
        .allocated = false,
        .enabled = true,
        .writer = ManifestRegionWriter::NONE,
    };
}

// Returns a ManifestRouterRegion that is backed by a particular L1 memory location.
ManifestRouterRegion make_l1_region(
    std::string id,
    std::string parent,
    ManifestRegionBacking backing,
    size_t address,
    size_t size,
    bool allocated,
    bool enabled,
    ManifestRegionWriter writer,
    std::string schema) {
    return {
        .id = std::move(id),
        .parent = std::move(parent),
        .backing = backing,
        .address = narrow_to_u32(address, "address"),
        .size = narrow_to_u32(size, "size"),
        .allocated = allocated,
        .enabled = enabled,
        .writer = writer,
        .schema = std::move(schema),
    };
}

// Returns a ManifestRouterRegion that is backed by a particular stream register.
ManifestRouterRegion make_stream_region(
    std::string id,
    std::string parent,
    uint32_t stream_id,
    bool enabled,
    ManifestRegionWriter writer,
    std::string schema = "stream_remote_dest_buf_space_available") {
    const bool allocated = stream_id != k_unused_stream_id;
    return {
        .id = std::move(id),
        .parent = std::move(parent),
        .backing = ManifestRegionBacking::STREAM_REG,
        .stream_id = allocated ? std::optional<uint32_t>(stream_id) : std::nullopt,
        .allocated = allocated,
        .enabled = allocated && enabled,
        .writer = writer,
        .schema = std::move(schema),
    };
}

// Returns the writer for a particular channel, based on the named arguments for the RISC that services it.
ManifestRegionWriter writer_for_channel(const std::vector<NamedArgs>& args, std::string_view prefix, uint32_t channel) {
    bool risc0 = false;
    bool risc1 = false;
    for (size_t risc = 0; risc < args.size(); ++risc) {
        const bool serviced = get_named_arg(args[risc], fmt::format("{}_{}_SERVICED", prefix, channel)) != 0;
        risc0 |= risc == 0 && serviced;
        risc1 |= risc == 1 && serviced;
    }
    if (risc0 && risc1) {
        return ManifestRegionWriter::ANY_ERISC;
    }
    if (risc0) {
        return ManifestRegionWriter::ERISC0;
    }
    if (risc1) {
        return ManifestRegionWriter::ERISC1;
    }
    return ManifestRegionWriter::NONE;
}

// Returns true if *vc* uses counters in l1 for completon credits / ack tracking, as opposed to stream registers.
bool vc_uses_l1_counters(const CreditTransportPlan& plan, uint32_t vc) {
    switch (vc) {
        case 0: return plan.vc0_uses_counters;
        case 1: return plan.vc1_uses_counters;
        case 2: return plan.vc2_uses_counters;
        default: TT_FATAL(false, "Invalid VC {}", vc); return false;
    }
}

// Adds padding regions to *regions* to fill gaps in the UNRESERVED L1 memory region.
void add_padding(std::vector<ManifestRouterRegion>& regions, uint32_t unreserved_base, uint32_t unreserved_size) {
    std::vector<std::pair<uint32_t, uint32_t>> intervals;
    const uint64_t unreserved_end = static_cast<uint64_t>(unreserved_base) + unreserved_size;
    for (const auto& region : regions) {
        if (region.backing != ManifestRegionBacking::UNRESERVED_L1 || !region.allocated || region.size == 0) {
            continue;
        }
        const uint64_t end = static_cast<uint64_t>(region.address) + region.size;
        TT_FATAL(
            region.address >= unreserved_base && end <= unreserved_end,
            "Debug region {} [{}, {}) lies outside UNRESERVED [{}, {})",
            region.id,
            region.address,
            end,
            unreserved_base,
            unreserved_end);
        intervals.emplace_back(region.address, static_cast<uint32_t>(end));
    }
    std::ranges::sort(intervals);

    uint32_t cursor = unreserved_base;
    uint32_t padding_index = 0;
    auto add_gap = [&](uint32_t begin, uint32_t end) {
        if (begin == end) {
            return;
        }
        regions.push_back(make_l1_region(
            fmt::format("padding.{}", padding_index++),
            "padding",
            ManifestRegionBacking::UNRESERVED_L1,
            begin,
            end - begin,
            true,
            false,
            ManifestRegionWriter::NONE,
            "raw"));
    };
    for (const auto& [begin, end] : intervals) {
        if (begin > cursor) {
            add_gap(cursor, begin);
        }
        cursor = std::max(cursor, end);
    }
    add_gap(cursor, static_cast<uint32_t>(unreserved_end));
}

}  // namespace

// Build a ManifestRouterInstance for a particular fabric router instance.
ManifestRouterInstance build_manifest_router_instance(
    const FabricEriscDatamoverBuilder& builder,
    const StreamAssignment& streams,
    const std::vector<std::unordered_map<std::string, uint32_t>>& named_ct_args_per_risc,
    const RouterLocation& location) {
    TT_FATAL(!named_ct_args_per_risc.empty(), "A fabric router debug layout requires at least one RISC argument map");
    TT_FATAL(
        named_ct_args_per_risc.size() == builder.get_configured_risc_count(),
        "Expected {} RISC argument maps, got {}",
        builder.get_configured_risc_count(),
        named_ct_args_per_risc.size());

    const auto& args = named_ct_args_per_risc.front();
    const auto& config = builder.config;
    const auto* allocator = dynamic_cast<const FabricStaticSizedChannelsAllocator*>(config.channel_allocator.get());
    TT_FATAL(allocator != nullptr, "Fabric router debug layout requires FabricStaticSizedChannelsAllocator");

    ManifestRouterInstance instance{
        .local_node = builder.local_fabric_node_id,
        .eth_chan = narrow_to_u32(builder.my_eth_channel, "eth_chan"),
        .peer_node = builder.peer_fabric_node_id,
        .direction = location.direction,
        .is_inter_mesh = builder.is_inter_mesh,
        .is_dispatch_link = location.is_dispatch_link,
        .num_active_eriscs = narrow_to_u32(builder.get_configured_risc_count(), "num_active_eriscs"),
        .sender_channels_per_vc = builder.get_actual_sender_channels_per_vc(),
        .receiver_channels_per_vc = builder.get_actual_receiver_channels_per_vc(),
        .worker_sender_channel = get_worker_connected_sender_channel(),
        .sender_producers =
            collect_sender_producer_names(builder.get_direction(), builder.get_actual_sender_channels_per_vc(), allocator),
        .credit_plan = streams.plan(),
        .first_level_ack_vc0 = get_named_arg(args, "ENABLE_FIRST_LEVEL_ACK_VC0") != 0,
        .downstream_edm_mask_vc0 = builder.get_downstream_edm_mask_for_vc(0),
        .downstream_edm_mask_vc1 =
            config.num_used_receiver_channels_per_vc[1] > 0 ? builder.get_downstream_edm_mask_for_vc(1) : 0,
        .downstream_edges_vc0 = collect_manifest_downstream_edges(builder, 0),
        .downstream_edges_vc1 = config.num_used_receiver_channels_per_vc[1] > 0
            ? collect_manifest_downstream_edges(builder, 1)
            : std::vector<ManifestDownstreamEdge>{},
        .has_tensix_extension = builder.has_tensix_extension_enabled(),
        .udm_mode = builder.is_udm_mode(),
    };
    auto& regions = instance.layout.regions;

    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const auto arch = hal.get_arch();
    const uint32_t unreserved_base = narrow_to_u32(
        hal.get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED),
        "UNRESERVED base");
    const uint32_t unreserved_size = narrow_to_u32(
        hal.get_dev_size(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED),
        "UNRESERVED size");

    regions.push_back(make_group_region("lifecycle"));
    require_named_address(args, "HANDSHAKE_ADDR", builder.get_handshake_address());
    auto handshake = make_l1_region(
        "lifecycle.handshake",
        "lifecycle",
        ManifestRegionBacking::UNRESERVED_L1,
        get_named_arg(args, "HANDSHAKE_ADDR"),
        FabricEriscDatamoverConfig::eth_channel_sync_size,
        true,
        true,
        ManifestRegionWriter::ANY_ERISC,
        "handshake_info_t");
    if (config.perf_telemetry_buffer_address != 0 && handshake.address < config.perf_telemetry_buffer_address + 32 &&
        config.perf_telemetry_buffer_address < handshake.address + handshake.size) {
        handshake.overlaps.push_back("diagnostics.perf_telemetry");
    }
    regions.push_back(std::move(handshake));

    const std::array lifecycle = {
        std::tuple{"lifecycle.edm_status", "EDM_STATUS_PTR_ADDR", config.edm_status_address, "EDMStatus"},
        std::tuple{
            "lifecycle.termination_signal",
            "TERMINATION_SIGNAL_ADDR",
            config.termination_signal_address,
            "TerminationSignal"},
        std::tuple{"lifecycle.local_sync", "EDM_LOCAL_SYNC_PTR_ADDR", config.edm_local_sync_address, "u32"},
        std::tuple{
            "lifecycle.local_tensix_sync",
            "EDM_LOCAL_TENSIX_SYNC_PTR_ADDR",
            config.edm_local_tensix_sync_address,
            "u32"},
    };
    for (const auto& [id, arg_name, address, schema] : lifecycle) {
        require_named_address(args, arg_name, address);
        const bool enabled =
            std::string_view(id) != "lifecycle.local_tensix_sync" || instance.has_tensix_extension || instance.udm_mode;
        regions.push_back(make_l1_region(
            id,
            "lifecycle",
            ManifestRegionBacking::UNRESERVED_L1,
            address,
            FabricEriscDatamoverConfig::field_size,
            true,
            enabled,
            ManifestRegionWriter::ANY_ERISC,
            schema));
    }
    regions.push_back(make_l1_region(
        "lifecycle.heartbeat",
        "lifecycle",
        ManifestRegionBacking::FIXED_L1,
        arch == tt::ARCH::BLACKHOLE ? FABRIC_KERNEL_HEARTBEAT_ADDR_BLACKHOLE : FABRIC_KERNEL_HEARTBEAT_ADDR_WORMHOLE,
        sizeof(uint32_t),
        true,
        true,
        ManifestRegionWriter::ANY_ERISC,
        "heartbeat_word"));

    regions.push_back(make_group_region("diagnostics"));
    const auto diagnostics = config.get_telemetry_and_metadata_buffer_map();
    const std::array diagnostic_specs = {
        std::tuple{
            "diagnostics.perf_telemetry",
            diagnostics.perf_telemetry,
            get_named_arg(args, "PERF_TELEMETRY_MODE") != 0,
            "perf_telemetry"},
        std::tuple{
            "diagnostics.code_profiling",
            diagnostics.code_profiling,
            get_named_arg(args, "CODE_PROFILING_ENABLED_TIMERS") != 0,
            "code_profiling"},
        std::tuple{
            "diagnostics.trimming",
            diagnostics.channel_trimming_capture,
            get_named_arg(args, "ENABLE_CHANNEL_TRIMMING_RESOURCE_USAGE_CAPTURE") != 0,
            "channel_trimming"},
    };
    for (const auto& [id, diagnostic, enabled, schema] : diagnostic_specs) {
        auto region = make_l1_region(
            id,
            "diagnostics",
            ManifestRegionBacking::UNRESERVED_L1,
            diagnostic.l1_address,
            diagnostic.size_bytes,
            diagnostic.l1_address != 0,
            enabled,
            ManifestRegionWriter::ANY_ERISC,
            schema);
        if (std::string_view(id) == "diagnostics.perf_telemetry" && diagnostic.l1_address != 0 &&
            builder.get_handshake_address() < diagnostic.l1_address + diagnostic.size_bytes &&
            diagnostic.l1_address <
                builder.get_handshake_address() + FabricEriscDatamoverConfig::eth_channel_sync_size) {
            region.overlaps.push_back("lifecycle.handshake");
        }
        regions.push_back(std::move(region));
    }

    regions.push_back(make_group_region("credits"));
    const std::array counter_specs = {
        std::tuple{
            "credits.to_sender_ack",
            "TO_SENDER_REMOTE_ACK_COUNTERS_BASE_ADDR",
            config.to_sender_channel_remote_ack_counters_base_addr,
            ManifestRegionWriter::PEER},
        std::tuple{
            "credits.to_sender_completion",
            "TO_SENDER_REMOTE_COMPLETION_COUNTERS_BASE_ADDR",
            config.to_sender_channel_remote_completion_counters_base_addr,
            ManifestRegionWriter::PEER},
        std::tuple{
            "credits.receiver_ack",
            "LOCAL_RECEIVER_ACK_COUNTERS_BASE_ADDR",
            config.receiver_channel_remote_ack_counters_base_addr,
            ManifestRegionWriter::ANY_ERISC},
        std::tuple{
            "credits.receiver_completion",
            "LOCAL_RECEIVER_COMPLETION_COUNTERS_BASE_ADDR",
            config.receiver_channel_remote_completion_counters_base_addr,
            ManifestRegionWriter::ANY_ERISC},
    };
    for (const auto& [id, arg_name, address, writer] : counter_specs) {
        require_named_address(args, arg_name, address);
        auto region = make_l1_region(
            id,
            "credits",
            ManifestRegionBacking::UNRESERVED_L1,
            address,
            config.router_buffer_clear_size_words,
            true,
            streams.plan().any_vc_uses_counters(),
            writer,
            "u32_counter_array");
        region.count = narrow_to_u32(config.num_used_sender_channels, "counter count");
        region.stride = sizeof(uint32_t);
        regions.push_back(std::move(region));
    }

    regions.push_back(make_group_region("sender"));
    for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        for (uint32_t channel = 0; channel < allocator->get_num_sender_channels(vc); ++channel) {
            const uint32_t flat = streams.sender_flat_base(vc) + channel;
            const bool enabled = channel < instance.sender_channels_per_vc[vc];
            const auto parent = fmt::format("sender.{}", flat);
            regions.push_back(make_group_region(parent, "sender"));
            const auto writer = writer_for_channel(named_ct_args_per_risc, "IS_SENDER_CHANNEL", flat);

            const std::array controls = {
                std::tuple{
                    "buffer_index",
                    config.sender_channels_buffer_index_address[flat],
                    FabricEriscDatamoverConfig::field_size,
                    "u32"},
                std::tuple{
                    "conn_info",
                    config.sender_channels_worker_conn_info_base_address[flat],
                    sizeof(EDMChannelWorkerLocationInfo),
                    "EDMChannelWorkerLocationInfo"},
                std::tuple{
                    "flow_semaphore",
                    config.sender_channels_local_flow_control_semaphore_address[flat],
                    FabricEriscDatamoverConfig::field_size,
                    "u32"},
                std::tuple{
                    "termination_status",
                    config.sender_channels_producer_terminate_connection_address[flat],
                    FabricEriscDatamoverConfig::field_size,
                    "u32"},
                std::tuple{
                    "connection",
                    config.sender_channels_connection_semaphore_address[flat],
                    FabricEriscDatamoverConfig::field_size,
                    "u32"},
                std::tuple{
                    "buffer_index_sem",
                    config.sender_channels_buffer_index_semaphore_address[flat],
                    FabricEriscDatamoverConfig::field_size,
                    "SenderChannelProducerCursor"},
            };
            require_named_address(
                args,
                fmt::format("LOCAL_SENDER_CH_{}_CONN_INFO_ADDR", flat),
                config.sender_channels_worker_conn_info_base_address[flat]);
            for (const auto& [name, address, size, schema] : controls) {
                regions.push_back(make_l1_region(
                    fmt::format("{}.control.{}", parent, name),
                    parent,
                    ManifestRegionBacking::UNRESERVED_L1,
                    address,
                    size,
                    true,
                    enabled,
                    writer,
                    schema));
            }

            auto ring = make_l1_region(
                fmt::format("{}.ring", parent),
                parent,
                ManifestRegionBacking::UNRESERVED_L1,
                allocator->get_sender_channel_base_address(vc, channel),
                allocator->get_sender_channel_number_of_slots(vc, channel) * config.channel_buffer_size_bytes,
                true,
                enabled,
                writer,
                "packet_ring");
            ring.count = narrow_to_u32(allocator->get_sender_channel_number_of_slots(vc, channel), "sender ring count");
            ring.stride = narrow_to_u32(config.channel_buffer_size_bytes, "sender ring stride");
            regions.push_back(std::move(ring));

            regions.push_back(make_stream_region(
                fmt::format("{}.free_slots", parent),
                parent,
                get_named_arg(args, fmt::format("SENDER_CHANNEL_{}_FREE_SLOTS_STREAM_ID", flat)),
                enabled,
                ManifestRegionWriter::WORKER));

            const bool counters = vc_uses_l1_counters(streams.plan(), vc);
            for (const auto& [suffix, arg_pattern, base, declared_extent] :
                 {std::tuple{
                      "acked",
                      "TO_SENDER_{}_PKTS_ACKED_ID",
                      config.to_sender_channel_remote_ack_counters_base_addr,
                      num_pkts_acked_names},
                  std::tuple{
                      "completed",
                      "TO_SENDER_{}_PKTS_COMPLETED_ID",
                      config.to_sender_channel_remote_completion_counters_base_addr,
                      num_pkts_completed_names}}) {
                const bool role_enabled =
                    enabled && (std::string_view(suffix) != "acked" || instance.first_level_ack_vc0);
                if (counters) {
                    auto region = make_l1_region(
                        fmt::format("{}.credits.{}", parent, suffix),
                        parent,
                        ManifestRegionBacking::UNRESERVED_L1,
                        base + flat * sizeof(uint32_t),
                        sizeof(uint32_t),
                        true,
                        role_enabled,
                        ManifestRegionWriter::PEER,
                        "u32_counter");
                    region.overlaps.push_back(
                        std::string_view(suffix) == "acked" ? "credits.to_sender_ack" : "credits.to_sender_completion");
                    regions.push_back(std::move(region));
                } else {
                    const uint32_t stream_id = flat < declared_extent
                                                   ? get_named_arg(args, fmt::format(fmt::runtime(arg_pattern), flat))
                                                   : k_unused_stream_id;
                    regions.push_back(make_stream_region(
                        fmt::format("{}.credits.{}", parent, suffix),
                        parent,
                        stream_id,
                        role_enabled,
                        ManifestRegionWriter::PEER));
                }
            }
        }
    }

    regions.push_back(make_group_region("receiver"));
    uint32_t receiver_flat = 0;
    for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
        for (uint32_t channel = 0; channel < allocator->get_num_receiver_channels(vc); ++channel, ++receiver_flat) {
            const bool enabled = channel < instance.receiver_channels_per_vc[vc];
            const auto parent = fmt::format("receiver.{}", receiver_flat);
            regions.push_back(make_group_region(parent, "receiver"));
            const auto writer = writer_for_channel(named_ct_args_per_risc, "IS_RECEIVER_CHANNEL", receiver_flat);
            auto ring = make_l1_region(
                fmt::format("{}.ring", parent),
                parent,
                ManifestRegionBacking::UNRESERVED_L1,
                allocator->get_receiver_channel_base_address(vc, channel),
                allocator->get_receiver_channel_number_of_slots(vc, channel) * config.channel_buffer_size_bytes,
                true,
                enabled,
                writer,
                "packet_ring");
            ring.count = narrow_to_u32(allocator->get_receiver_channel_number_of_slots(vc, channel), "receiver ring count");
            ring.stride = narrow_to_u32(config.channel_buffer_size_bytes, "receiver ring stride");
            regions.push_back(std::move(ring));
            regions.push_back(make_stream_region(
                fmt::format("{}.pkts_sent", parent),
                parent,
                receiver_flat < num_receiver_pkts_sent_names
                    ? get_named_arg(args, fmt::format("TO_RECEIVER_{}_PKTS_SENT_ID", receiver_flat))
                    : k_unused_stream_id,
                enabled,
                writer));
        }
    }
    for (uint32_t channel = 0; channel < builder_config::max_downstream_edms; ++channel) {
        const size_t address = config.receiver_channels_downstream_teardown_semaphore_address[channel];
        const bool allocated = address != 0;
        const bool enabled = allocated && channel < config.num_used_receiver_channels;
        regions.push_back(make_l1_region(
            fmt::format("receiver.downstream_teardown_sem.{}", channel),
            "receiver",
            ManifestRegionBacking::UNRESERVED_L1,
            address,
            FabricEriscDatamoverConfig::field_size,
            allocated,
            enabled,
            ManifestRegionWriter::ANY_ERISC,
            "u32"));
    }

    for (uint32_t vc = 0; vc < 2; ++vc) {
        const uint32_t mask = vc == 0 ? instance.downstream_edm_mask_vc0 : instance.downstream_edm_mask_vc1;
        for (uint32_t edge = 1; edge <= 4; ++edge) {
            regions.push_back(make_stream_region(
                fmt::format("credits.downstream.vc{}.edge{}.free_slots", vc, edge),
                "credits",
                get_named_arg(args, fmt::format("VC{}_FREE_SLOTS_FROM_DOWNSTREAM_EDGE_{}_STREAM_ID", vc, edge)),
                (mask & (1U << (edge - 1))) != 0,
                ManifestRegionWriter::PEER));
        }
    }
    regions.push_back(make_stream_region(
        "credits.vc2_receiver.free_slots",
        "credits",
        get_named_arg(args, "VC2_RECEIVER_FREE_SLOTS_STREAM_ID"),
        instance.sender_channels_per_vc[2] != 0,
        ManifestRegionWriter::PEER));
    regions.push_back(make_stream_region(
        "credits.tensix_relay.free_slots",
        "credits",
        get_named_arg(args, "TENSIX_RELAY_LOCAL_FREE_SLOTS_STREAM_ID"),
        instance.udm_mode,
        ManifestRegionWriter::WORKER));

    regions.push_back(make_group_region("relay"));
    regions.push_back(make_l1_region(
        "relay.connection_buffer_index",
        "relay",
        ManifestRegionBacking::UNRESERVED_L1,
        config.tensix_relay_connection_buffer_index_id,
        FabricEriscDatamoverConfig::field_size,
        true,
        instance.udm_mode,
        ManifestRegionWriter::ANY_ERISC,
        "u32"));

    if (arch == tt::ARCH::BLACKHOLE) {
        require_named_address(
            args,
            "NOTIFY_WORKER_OF_READ_COUNTER_UPDATE_SRC_ADDR",
            config.notify_worker_of_read_counter_update_src_address);
        regions.push_back(make_group_region("misc"));
        regions.push_back(make_l1_region(
            "misc.notify_worker_src",
            "misc",
            ManifestRegionBacking::UNRESERVED_L1,
            config.notify_worker_of_read_counter_update_src_address,
            FabricEriscDatamoverConfig::field_size,
            true,
            true,
            ManifestRegionWriter::ANY_ERISC,
            "u32"));
    }

    regions.push_back(make_group_region("unused"));
    if (config.handshake_addr != builder.get_handshake_address()) {
        regions.push_back(make_l1_region(
            "unused.config_handshake",
            "unused",
            ManifestRegionBacking::UNRESERVED_L1,
            config.handshake_addr,
            FabricEriscDatamoverConfig::eth_channel_sync_size,
            true,
            false,
            ManifestRegionWriter::NONE,
            "raw"));
    }
    regions.push_back(make_l1_region(
        "unused.edm_channel_ack",
        "unused",
        ManifestRegionBacking::UNRESERVED_L1,
        config.edm_channel_ack_addr,
        4 * FabricEriscDatamoverConfig::eth_channel_sync_size,
        true,
        false,
        ManifestRegionWriter::NONE,
        "raw"));

    regions.push_back(make_group_region("hal"));
    for (const auto& [id, type, schema] :
         {std::tuple{"hal.telemetry", tt::tt_metal::HalL1MemAddrType::FABRIC_TELEMETRY, "fabric_telemetry"},
          std::tuple{"hal.routing_table", tt::tt_metal::HalL1MemAddrType::ROUTING_TABLE, "routing_l1_info_t"}}) {
        regions.push_back(make_l1_region(
            id,
            "hal",
            ManifestRegionBacking::FIXED_L1,
            hal.get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, type),
            hal.get_dev_size(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, type),
            true,
            true,
            ManifestRegionWriter::HOST,
            schema));
    }

    regions.push_back(make_group_region("padding"));
    add_padding(regions, unreserved_base, unreserved_size);
    return instance;
}

}  // namespace tt::tt_fabric
