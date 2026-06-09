#include "fabric_mux_saturation_program.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "context/metal_context.hpp"
#include "fabric_mux_benchmark_program_utils.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"

namespace tt::tt_fabric::bench {

namespace {

constexpr char kV2SaturationSenderKernelPath[] =
    "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_mux_saturation_v2_sender.cpp";
constexpr char kV1SaturationSenderKernelPath[] =
    "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_mux_saturation_v1_sender.cpp";
constexpr char kV1MuxKernelPath[] = "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp";

constexpr uint32_t kTestResultsSizeBytes = 128;
constexpr uint32_t kPacketHeaderBufferSizeBytes = 1024;
constexpr uint32_t kNocAddressPaddingBytes = 16;
constexpr uint32_t kSenderStartDelayCycles = 100000;
constexpr std::array<tt::tt_fabric::RoutingDirection, 4> kRoutingDirections = {
    tt::tt_fabric::RoutingDirection::N,
    tt::tt_fabric::RoutingDirection::S,
    tt::tt_fabric::RoutingDirection::E,
    tt::tt_fabric::RoutingDirection::W,
};

constexpr auto kDefaultV2ForwarderNoc = tt::tt_metal::NOC::RISCV_0_default;

struct SenderMemoryMap {
    uint32_t test_results_address = 0;
    uint32_t start_signal_address = 0;
    uint32_t ready_count_address = 0;
    uint32_t local_mux_status_address = 0;
    uint32_t local_flow_control_address = 0;
    uint32_t local_teardown_address = 0;
    uint32_t local_buffer_index_address = 0;
    uint32_t packet_header_buffer_address = 0;
    uint32_t payload_buffer_address = 0;
    uint32_t end_address = 0;
};

struct PassiveTarget {
    CoreCoord logical_core;
    CoreCoord virtual_core;
    uint32_t target_address = 0;
    uint32_t linear_num_hops = 1;
    uint32_t dst_device_id = 0;
    uint32_t dst_mesh_id = 0;
};

struct SaturationPlacement {
    MeshDevicePtr sender_device;
    tt::tt_fabric::FabricNodeId src_fabric_node_id;
    tt::tt_fabric::FabricNodeId next_hop_fabric_node_id;
    CoreCoord mux_logical_core;
    CoreCoord mux_virtual_core;
    std::vector<CoreCoord> sender_logical_cores;
    std::vector<SenderMemoryMap> sender_memory_maps;
    std::vector<PassiveTarget> passive_targets;
    std::vector<uint32_t> sender_noc_xy_encodings;
};

struct PreparedSaturationRun {
    SaturationPlacement placement;
    uint32_t packet_payload_size_bytes = 0;
    uint64_t expected_sender_bytes = 0;
    uint64_t expected_aggregate_bytes = 0;
};

ChipId get_physical_device_id(const MeshDevicePtr& device) { return device->get_devices()[0]->id(); }

tt::tt_fabric::FabricConfig to_fabric_config(SaturationTopology topology) {
    switch (topology) {
        case SaturationTopology::Fabric1D: return tt::tt_fabric::FabricConfig::FABRIC_1D;
        case SaturationTopology::Fabric2D: return tt::tt_fabric::FabricConfig::FABRIC_2D;
    }
    TT_THROW("Unhandled saturation topology");
}

uint32_t get_l1_alignment() {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    return static_cast<uint32_t>(hal.get_alignment(tt::tt_metal::HalMemType::L1));
}

std::optional<uint32_t> resolve_packet_payload_size_bytes(
    const SaturationCase& benchmark_case, std::string* rejection_reason = nullptr) {
    const auto packet_header_size_bytes =
        static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_packet_header_size_bytes());
    if (benchmark_case.channel_buffer_size_bytes <= packet_header_size_bytes) {
        if (rejection_reason != nullptr) {
            std::ostringstream message;
            message << "channel_buffer_size_bytes must exceed fabric header size " << packet_header_size_bytes
                    << ", got " << benchmark_case.channel_buffer_size_bytes;
            *rejection_reason = message.str();
        }
        return std::nullopt;
    }

    const auto max_channel_buffer_size_bytes =
        static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes());
    if (benchmark_case.channel_buffer_size_bytes > max_channel_buffer_size_bytes) {
        if (rejection_reason != nullptr) {
            std::ostringstream message;
            message << "channel_buffer_size_bytes exceeds fabric max " << max_channel_buffer_size_bytes << ", got "
                    << benchmark_case.channel_buffer_size_bytes;
            *rejection_reason = message.str();
        }
        return std::nullopt;
    }

    const auto packet_payload_size_bytes = benchmark_case.channel_buffer_size_bytes - packet_header_size_bytes;
    if ((packet_payload_size_bytes % 16) != 0) {
        if (rejection_reason != nullptr) {
            std::ostringstream message;
            message << "resolved payload size must be a multiple of 16 bytes, got " << packet_payload_size_bytes;
            *rejection_reason = message.str();
        }
        return std::nullopt;
    }

    return packet_payload_size_bytes;
}

SenderMemoryMap create_sender_memory_map(uint32_t base_l1_address, uint32_t packet_payload_size_bytes) {
    SenderMemoryMap memory_map{};
    uint32_t current_address = base_l1_address;

    memory_map.test_results_address = current_address;
    current_address += kTestResultsSizeBytes;

    current_address = align_up(current_address, kNocAddressPaddingBytes);
    memory_map.start_signal_address = current_address;
    current_address += kNocAddressPaddingBytes;

    current_address = align_up(current_address, kNocAddressPaddingBytes);
    memory_map.ready_count_address = current_address;
    current_address += kNocAddressPaddingBytes;

    current_address = align_up(current_address, kNocAddressPaddingBytes);
    memory_map.local_mux_status_address = current_address;
    current_address += kNocAddressPaddingBytes;

    current_address = align_up(current_address, kNocAddressPaddingBytes);
    memory_map.local_flow_control_address = current_address;
    current_address += kNocAddressPaddingBytes;

    current_address = align_up(current_address, kNocAddressPaddingBytes);
    memory_map.local_teardown_address = current_address;
    current_address += kNocAddressPaddingBytes;

    current_address = align_up(current_address, kNocAddressPaddingBytes);
    memory_map.local_buffer_index_address = current_address;
    current_address += kNocAddressPaddingBytes;

    current_address = align_up(current_address, kNocAddressPaddingBytes);
    memory_map.packet_header_buffer_address = current_address;
    current_address += kPacketHeaderBufferSizeBytes;

    current_address = align_up(current_address, kNocAddressPaddingBytes);
    memory_map.payload_buffer_address = current_address;
    current_address += packet_payload_size_bytes;

    memory_map.end_address = current_address;
    return memory_map;
}

bool validate_sender_memory_map(const SenderMemoryMap& memory_map, std::string* rejection_reason = nullptr) {
    if (memory_map.end_address <= get_worker_l1_end_address()) {
        return true;
    }
    if (rejection_reason != nullptr) {
        std::ostringstream message;
        message << "sender memory map end address " << memory_map.end_address << " exceeds worker L1 end "
                << get_worker_l1_end_address();
        *rejection_reason = message.str();
    }
    return false;
}

void initialize_saturation_sender_state(
    tt::tt_metal::IDevice* device,
    const CoreCoord& sender_logical_core,
    const SenderMemoryMap& sender_memory_map,
    uint32_t initial_ready_count) {
    initialize_sender_start_barrier_state(
        device,
        sender_logical_core,
        sender_memory_map.start_signal_address,
        sender_memory_map.ready_count_address,
        initial_ready_count);
    write_word_to_device(device, sender_logical_core, sender_memory_map.local_mux_status_address, 0);
    write_word_to_device(device, sender_logical_core, sender_memory_map.local_flow_control_address, 0);
    write_word_to_device(device, sender_logical_core, sender_memory_map.local_teardown_address, 0);
    write_word_to_device(device, sender_logical_core, sender_memory_map.local_buffer_index_address, 0);
}

std::optional<std::vector<SenderMemoryMap>> build_sender_memory_maps(
    const MeshDevicePtr& sender_device,
    std::size_t num_senders,
    uint32_t packet_payload_size_bytes,
    std::string* rejection_reason = nullptr) {
    const auto sender_l1_base_address =
        static_cast<uint32_t>(sender_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1));

    std::vector<SenderMemoryMap> sender_memory_maps;
    sender_memory_maps.reserve(num_senders);
    for (std::size_t sender_idx = 0; sender_idx < num_senders; ++sender_idx) {
        const auto memory_map = create_sender_memory_map(sender_l1_base_address, packet_payload_size_bytes);
        std::string sender_memory_reason;
        if (!validate_sender_memory_map(memory_map, &sender_memory_reason)) {
            if (rejection_reason != nullptr) {
                *rejection_reason = std::move(sender_memory_reason);
            }
            return std::nullopt;
        }
        sender_memory_maps.push_back(memory_map);
    }

    return sender_memory_maps;
}

std::optional<std::vector<PassiveTarget>> build_passive_targets(
    const MeshDevicePtr& remote_device,
    const tt::tt_fabric::FabricNodeId& next_hop_fabric_node_id,
    std::size_t num_senders,
    uint32_t packet_payload_size_bytes,
    std::string* rejection_reason = nullptr) {
    auto remote_worker_cores = enumerate_worker_cores(remote_device);
    if (remote_worker_cores.empty()) {
        if (rejection_reason != nullptr) {
            *rejection_reason = "next-hop remote device did not expose any worker cores";
        }
        return std::nullopt;
    }

    const auto l1_alignment = get_l1_alignment();
    const auto remote_target_base_address = align_up(
        static_cast<uint32_t>(remote_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1)),
        l1_alignment);
    std::vector<uint32_t> next_target_address_by_core(remote_worker_cores.size(), remote_target_base_address);
    std::vector<PassiveTarget> passive_targets;
    passive_targets.reserve(num_senders);

    for (std::size_t sender_idx = 0; sender_idx < num_senders; ++sender_idx) {
        const std::size_t remote_core_idx = sender_idx % remote_worker_cores.size();
        const auto target_address = next_target_address_by_core[remote_core_idx];
        const auto next_target_address = align_up(target_address + packet_payload_size_bytes, l1_alignment);
        if (next_target_address > get_worker_l1_end_address()) {
            if (rejection_reason != nullptr) {
                std::ostringstream message;
                message << "remote passive target regions exceed worker L1 on device "
                        << get_physical_device_id(remote_device);
                *rejection_reason = message.str();
            }
            return std::nullopt;
        }

        const auto& target_logical_core = remote_worker_cores[remote_core_idx];
        const auto target_virtual_core = remote_device->worker_core_from_logical_core(target_logical_core);
        passive_targets.push_back(PassiveTarget{
            .logical_core = target_logical_core,
            .virtual_core = target_virtual_core,
            .target_address = target_address,
            .linear_num_hops = 1,
            .dst_device_id = static_cast<uint32_t>(next_hop_fabric_node_id.chip_id),
            .dst_mesh_id = static_cast<uint32_t>(*next_hop_fabric_node_id.mesh_id),
        });
        next_target_address_by_core[remote_core_idx] = next_target_address;
    }

    return passive_targets;
}

void create_v2_sender_kernel(
    tt::tt_metal::Program& program,
    const CoreCoord& sender_logical_core,
    const SenderMemoryMap& sender_memory_map,
    const tt::tt_fabric::FabricMuxV2Config& mux_config,
    const CoreCoord& mux_virtual_core,
    uint32_t logical_channel_id,
    uint32_t packet_payload_size_bytes,
    uint32_t num_packets_per_sender,
    const PassiveTarget& passive_target,
    uint32_t seed,
    bool is_master_sender,
    uint32_t master_sender_noc_xy_encoding,
    uint32_t expected_ready_count,
    const std::vector<uint32_t>& peer_sender_noc_xy_encodings,
    bool is_2d_fabric) {
    const auto flow_control_sem_id = tt::tt_metal::CreateSemaphore(program, sender_logical_core, 0);
    const auto teardown_sem_id = tt::tt_metal::CreateSemaphore(program, sender_logical_core, 0);

    const std::vector<uint32_t> compile_args = {
        sender_memory_map.test_results_address,
        kTestResultsSizeBytes,
        sender_memory_map.start_signal_address,
        sender_memory_map.ready_count_address,
        static_cast<uint32_t>(is_master_sender),
        static_cast<uint32_t>(is_2d_fabric),
    };

    std::vector<uint32_t> runtime_args = {
        sender_memory_map.packet_header_buffer_address,
        sender_memory_map.payload_buffer_address,
        packet_payload_size_bytes,
        num_packets_per_sender,
        seed,
        kSenderStartDelayCycles,
        static_cast<uint32_t>(passive_target.virtual_core.x),
        static_cast<uint32_t>(passive_target.virtual_core.y),
        passive_target.target_address,
        passive_target.linear_num_hops,
        passive_target.dst_device_id,
        passive_target.dst_mesh_id,
        master_sender_noc_xy_encoding,
        expected_ready_count,
        static_cast<uint32_t>(peer_sender_noc_xy_encodings.size()),
    };
    runtime_args.insert(runtime_args.end(), peer_sender_noc_xy_encodings.begin(), peer_sender_noc_xy_encodings.end());

    mux_config.append_client_connection_rt_args(
        mux_virtual_core,
        static_cast<uint8_t>(logical_channel_id),
        tt::tt_fabric::FabricMuxV2Config::ClientSemaphores{
            .flow_control_sem_id = flow_control_sem_id,
            .teardown_sem_id = teardown_sem_id,
        },
        runtime_args);

    create_data_movement_kernel(
        program, kV2SaturationSenderKernelPath, sender_logical_core, compile_args, runtime_args);
}

void create_v1_sender_kernel(
    tt::tt_metal::Program& program,
    const CoreCoord& sender_logical_core,
    const SenderMemoryMap& sender_memory_map,
    const tt::tt_fabric::FabricMuxConfig& mux_config,
    const CoreCoord& mux_virtual_core,
    uint32_t logical_channel_id,
    uint32_t packet_payload_size_bytes,
    uint32_t num_packets_per_sender,
    const PassiveTarget& passive_target,
    uint32_t seed,
    bool is_master_sender,
    uint32_t master_sender_noc_xy_encoding,
    uint32_t expected_ready_count,
    const std::vector<uint32_t>& peer_sender_noc_xy_encodings,
    bool is_2d_fabric) {
    constexpr auto kChannelType = tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL;
    const auto channel_id = static_cast<uint8_t>(logical_channel_id);

    const std::vector<uint32_t> compile_args = {
        sender_memory_map.test_results_address,
        kTestResultsSizeBytes,
        sender_memory_map.start_signal_address,
        sender_memory_map.ready_count_address,
        sender_memory_map.local_mux_status_address,
        sender_memory_map.local_flow_control_address,
        sender_memory_map.local_teardown_address,
        sender_memory_map.local_buffer_index_address,
        static_cast<uint32_t>(is_master_sender),
        static_cast<uint32_t>(is_2d_fabric),
        static_cast<uint32_t>(mux_virtual_core.x),
        static_cast<uint32_t>(mux_virtual_core.y),
        mux_config.get_num_buffers(kChannelType),
        to_uint32_checked(mux_config.get_buffer_size_bytes(kChannelType), "v1_channel_buffer_size_bytes"),
        to_uint32_checked(mux_config.get_channel_base_address(kChannelType, channel_id), "v1_channel_base_address"),
        to_uint32_checked(
            mux_config.get_connection_info_address(kChannelType, channel_id), "v1_connection_info_address"),
        to_uint32_checked(
            mux_config.get_connection_handshake_address(kChannelType, channel_id), "v1_connection_handshake_address"),
        to_uint32_checked(mux_config.get_flow_control_address(kChannelType, channel_id), "v1_flow_control_address"),
        to_uint32_checked(mux_config.get_buffer_index_address(kChannelType, channel_id), "v1_buffer_index_address"),
        to_uint32_checked(mux_config.get_status_address(), "v1_mux_status_address"),
        logical_channel_id,
    };

    std::vector<uint32_t> runtime_args = {
        sender_memory_map.packet_header_buffer_address,
        sender_memory_map.payload_buffer_address,
        packet_payload_size_bytes,
        num_packets_per_sender,
        seed,
        kSenderStartDelayCycles,
        static_cast<uint32_t>(passive_target.virtual_core.x),
        static_cast<uint32_t>(passive_target.virtual_core.y),
        passive_target.target_address,
        passive_target.linear_num_hops,
        passive_target.dst_device_id,
        passive_target.dst_mesh_id,
        master_sender_noc_xy_encoding,
        expected_ready_count,
        static_cast<uint32_t>(peer_sender_noc_xy_encodings.size()),
    };
    runtime_args.insert(runtime_args.end(), peer_sender_noc_xy_encodings.begin(), peer_sender_noc_xy_encodings.end());

    create_data_movement_kernel(
        program, kV1SaturationSenderKernelPath, sender_logical_core, compile_args, runtime_args);
}

SaturationRunResult make_error(std::string error_message) {
    SaturationRunResult result{};
    result.error_message = std::move(error_message);
    return result;
}

SaturationRunResult to_run_result(const SenderResultSummary& summary) {
    SaturationRunResult result{};
    result.success = summary.success;
    result.error_message = summary.error_message;
    result.aggregate_bytes = summary.aggregate_bytes;
    result.max_sender_cycles = summary.max_sender_cycles;
    return result;
}

std::vector<SenderExecutionContext> build_sender_execution_contexts(const SaturationPlacement& placement) {
    TT_FATAL(
        placement.sender_logical_cores.size() == placement.sender_memory_maps.size(),
        "Sender placement cores ({}) and memory maps ({}) must match",
        placement.sender_logical_cores.size(),
        placement.sender_memory_maps.size());
    std::vector<SenderExecutionContext> sender_execution_contexts;
    sender_execution_contexts.reserve(placement.sender_logical_cores.size());
    for (std::size_t sender_idx = 0; sender_idx < placement.sender_logical_cores.size(); ++sender_idx) {
        sender_execution_contexts.push_back(SenderExecutionContext{
            .logical_core = placement.sender_logical_cores[sender_idx],
            .test_results_address = placement.sender_memory_maps[sender_idx].test_results_address,
        });
    }
    return sender_execution_contexts;
}

std::optional<SaturationPlacement> select_next_hop_saturation_placement(
    const FabricMuxSaturationBenchmarkContext& context,
    const SaturationCase& benchmark_case,
    uint32_t packet_payload_size_bytes,
    std::string* rejection_reason = nullptr) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    for (const auto& sender_device : context.get_devices()) {
        auto sender_worker_cores = enumerate_worker_cores(sender_device);
        if (sender_worker_cores.size() < static_cast<std::size_t>(benchmark_case.num_clients + 1)) {
            continue;
        }

        const auto mux_logical_core = sender_worker_cores.front();
        sender_worker_cores.erase(sender_worker_cores.begin());
        std::vector<CoreCoord> sender_logical_cores(
            sender_worker_cores.begin(),
            sender_worker_cores.begin() + static_cast<std::ptrdiff_t>(benchmark_case.num_clients));

        const auto src_fabric_node_id =
            control_plane.get_fabric_node_id_from_physical_chip_id(get_physical_device_id(sender_device));
        for (const auto direction : kRoutingDirections) {
            const auto neighbors = control_plane.get_intra_chip_neighbors(src_fabric_node_id, direction);
            if (neighbors.empty()) {
                continue;
            }

            const auto next_hop_fabric_node_id = tt::tt_fabric::FabricNodeId(src_fabric_node_id.mesh_id, neighbors[0]);
            const auto next_hop_physical_chip_id =
                control_plane.get_physical_chip_id_from_fabric_node_id(next_hop_fabric_node_id);
            auto remote_device_it = context.get_devices_by_physical_id().find(next_hop_physical_chip_id);
            if (remote_device_it == context.get_devices_by_physical_id().end()) {
                continue;
            }

            const auto& remote_device = remote_device_it->second;
            auto sender_memory_maps = build_sender_memory_maps(
                sender_device, sender_logical_cores.size(), packet_payload_size_bytes, rejection_reason);
            if (!sender_memory_maps.has_value()) {
                continue;
            }

            auto passive_targets = build_passive_targets(
                remote_device,
                next_hop_fabric_node_id,
                sender_logical_cores.size(),
                packet_payload_size_bytes,
                rejection_reason);
            if (!passive_targets.has_value()) {
                continue;
            }

            auto sender_noc_xy_encodings = build_sender_noc_xy_encodings(sender_device, sender_logical_cores);
            return SaturationPlacement{
                .sender_device = sender_device,
                .src_fabric_node_id = src_fabric_node_id,
                .next_hop_fabric_node_id = next_hop_fabric_node_id,
                .mux_logical_core = mux_logical_core,
                .mux_virtual_core = sender_device->worker_core_from_logical_core(mux_logical_core),
                .sender_logical_cores = std::move(sender_logical_cores),
                .sender_memory_maps = std::move(sender_memory_maps.value()),
                .passive_targets = std::move(passive_targets.value()),
                .sender_noc_xy_encodings = std::move(sender_noc_xy_encodings),
            };
        }
    }

    if (rejection_reason != nullptr && rejection_reason->empty()) {
        *rejection_reason = "no source chip with enough sender cores and a reachable next-hop destination device";
    }
    return std::nullopt;
}

std::optional<PreparedSaturationRun> prepare_common_saturation_run(
    const FabricMuxSaturationBenchmarkContext& context,
    const SaturationCase& benchmark_case,
    std::string* rejection_reason = nullptr) {
    auto packet_payload_size_bytes = resolve_packet_payload_size_bytes(benchmark_case, rejection_reason);
    if (!packet_payload_size_bytes.has_value()) {
        return std::nullopt;
    }

    auto placement = select_next_hop_saturation_placement(
        context, benchmark_case, packet_payload_size_bytes.value(), rejection_reason);
    if (!placement.has_value()) {
        return std::nullopt;
    }

    const uint64_t expected_sender_bytes =
        static_cast<uint64_t>(benchmark_case.num_packets_per_sender) * packet_payload_size_bytes.value();
    return PreparedSaturationRun{
        .placement = std::move(placement.value()),
        .packet_payload_size_bytes = packet_payload_size_bytes.value(),
        .expected_sender_bytes = expected_sender_bytes,
        .expected_aggregate_bytes = expected_sender_bytes * benchmark_case.num_clients,
    };
}

SaturationRunResult run_v1_saturation_once(
    const FabricMuxSaturationBenchmarkContext& context, const SaturationCase& benchmark_case) {
    std::string placement_rejection_reason;
    auto prepared_run = prepare_common_saturation_run(context, benchmark_case, &placement_rejection_reason);
    if (!prepared_run.has_value()) {
        return make_error(placement_rejection_reason);
    }
    const auto& placement = prepared_run->placement;

    auto* sender_device_handle = placement.sender_device->get_devices()[0];

    const auto forwarding_links =
        tt::tt_fabric::get_forwarding_link_indices(placement.src_fabric_node_id, placement.next_hop_fabric_node_id);
    if (forwarding_links.empty()) {
        return make_error("no forwarding links available between source chip and next-hop destination");
    }

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    tt::tt_fabric::FabricMuxConfig mux_config(
        static_cast<uint8_t>(benchmark_case.num_clients),
        0,
        benchmark_case.num_buffers_per_channel,
        0,
        benchmark_case.channel_buffer_size_bytes,
        placement.sender_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1));
    const auto mux_compile_args = mux_config.get_fabric_mux_compile_time_args();
    const auto mux_runtime_args = mux_config.get_fabric_mux_run_time_args(
        placement.src_fabric_node_id,
        placement.next_hop_fabric_node_id,
        forwarding_links.front(),
        program,
        placement.mux_logical_core);
    create_data_movement_kernel(
        program, kV1MuxKernelPath, placement.mux_logical_core, mux_compile_args, mux_runtime_args);

    const auto master_sender_noc_xy_encoding = placement.sender_noc_xy_encodings.front();
    const auto expected_ready_count = static_cast<uint32_t>(placement.sender_logical_cores.size());
    for (std::size_t sender_idx = 0; sender_idx < placement.sender_logical_cores.size(); ++sender_idx) {
        initialize_saturation_sender_state(
            sender_device_handle,
            placement.sender_logical_cores[sender_idx],
            placement.sender_memory_maps[sender_idx],
            sender_idx == 0 ? 1u : 0u);

        auto peer_sender_noc_xy_encodings =
            build_peer_sender_noc_xy_encodings(placement.sender_noc_xy_encodings, sender_idx);

        create_v1_sender_kernel(
            program,
            placement.sender_logical_cores[sender_idx],
            placement.sender_memory_maps[sender_idx],
            mux_config,
            placement.mux_virtual_core,
            static_cast<uint32_t>(sender_idx),
            prepared_run->packet_payload_size_bytes,
            benchmark_case.num_packets_per_sender,
            placement.passive_targets[sender_idx],
            0x12340000u + static_cast<uint32_t>(sender_idx),
            sender_idx == 0,
            master_sender_noc_xy_encoding,
            expected_ready_count,
            peer_sender_noc_xy_encodings,
            context.get_topology() == SaturationTopology::Fabric2D);
    }

    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(sender_device_handle->id());

    enqueue_single_device_mesh_program(placement.sender_device, std::move(program));

    return to_run_result(read_and_validate_sender_results(
        sender_device_handle,
        build_sender_execution_contexts(placement),
        kTestResultsSizeBytes,
        prepared_run->expected_sender_bytes,
        prepared_run->expected_aggregate_bytes));
}

SaturationRunResult run_v2_saturation_once(
    const FabricMuxSaturationBenchmarkContext& context, const SaturationCase& benchmark_case) {
    std::string placement_rejection_reason;
    auto prepared_run = prepare_common_saturation_run(context, benchmark_case, &placement_rejection_reason);
    if (!prepared_run.has_value()) {
        return make_error(placement_rejection_reason);
    }
    const auto& placement = prepared_run->placement;

    auto* sender_device_handle = placement.sender_device->get_devices()[0];

    const auto forwarding_links =
        tt::tt_fabric::get_forwarding_link_indices(placement.src_fabric_node_id, placement.next_hop_fabric_node_id);
    if (forwarding_links.empty()) {
        return make_error("no forwarding links available between source chip and next-hop destination");
    }

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    tt::tt_fabric::FabricMuxV2Config mux_config(
        static_cast<uint8_t>(benchmark_case.num_clients),
        benchmark_case.num_buffers_per_channel,
        benchmark_case.channel_buffer_size_bytes,
        placement.sender_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1));
    tt::tt_fabric::add_fabric_mux_v2_to_program(
        program,
        mux_config,
        placement.mux_logical_core,
        placement.src_fabric_node_id,
        placement.next_hop_fabric_node_id,
        forwarding_links.front(),
        kDefaultV2ForwarderNoc);

    const auto master_sender_noc_xy_encoding = placement.sender_noc_xy_encodings.front();
    const auto expected_ready_count = static_cast<uint32_t>(placement.sender_logical_cores.size());
    for (std::size_t sender_idx = 0; sender_idx < placement.sender_logical_cores.size(); ++sender_idx) {
        initialize_saturation_sender_state(
            sender_device_handle,
            placement.sender_logical_cores[sender_idx],
            placement.sender_memory_maps[sender_idx],
            sender_idx == 0 ? 1u : 0u);

        auto peer_sender_noc_xy_encodings =
            build_peer_sender_noc_xy_encodings(placement.sender_noc_xy_encodings, sender_idx);

        create_v2_sender_kernel(
            program,
            placement.sender_logical_cores[sender_idx],
            placement.sender_memory_maps[sender_idx],
            mux_config,
            placement.mux_virtual_core,
            static_cast<uint32_t>(sender_idx),
            prepared_run->packet_payload_size_bytes,
            benchmark_case.num_packets_per_sender,
            placement.passive_targets[sender_idx],
            0x12340000u + static_cast<uint32_t>(sender_idx),
            sender_idx == 0,
            master_sender_noc_xy_encoding,
            expected_ready_count,
            peer_sender_noc_xy_encodings,
            context.get_topology() == SaturationTopology::Fabric2D);
    }

    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(sender_device_handle->id());

    enqueue_single_device_mesh_program(placement.sender_device, std::move(program));

    return to_run_result(read_and_validate_sender_results(
        sender_device_handle,
        build_sender_execution_contexts(placement),
        kTestResultsSizeBytes,
        prepared_run->expected_sender_bytes,
        prepared_run->expected_aggregate_bytes));
}

}  // namespace

void FabricMuxSaturationBenchmarkContext::initialize(SaturationTopology topology) {
    shutdown();

    topology_ = topology;
    tt::tt_fabric::SetFabricConfig(
        to_fabric_config(topology_), tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);

    const auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    std::vector<ChipId> ids;
    ids.reserve(num_devices);
    for (unsigned int id = 0; id < num_devices; ++id) {
        ids.push_back(id);
    }

    const auto& dispatch_core_config = tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
    devices_by_physical_id_ = tt::tt_metal::distributed::MeshDevice::create_unit_meshes(
        ids, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, dispatch_core_config, {}, DEFAULT_WORKER_L1_SIZE);
    devices_.reserve(devices_by_physical_id_.size());
    for (const auto& [id, device] : devices_by_physical_id_) {
        devices_.push_back(device);
    }
}

void FabricMuxSaturationBenchmarkContext::shutdown() {
    for (auto& [id, device] : devices_by_physical_id_) {
        if (device != nullptr) {
            [[maybe_unused]] const bool closed = device->close();
        }
    }
    devices_by_physical_id_.clear();
    devices_.clear();
    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
}

bool FabricMuxSaturationBenchmarkContext::can_support_case(
    const SaturationVariant& variant, const SaturationCase& benchmark_case, std::string* rejection_reason) const {
    if (benchmark_case.num_clients == 0) {
        if (rejection_reason != nullptr) {
            *rejection_reason = "num_clients must be greater than 0";
        }
        return false;
    }

    if (benchmark_case.num_clients > std::numeric_limits<uint8_t>::max()) {
        if (rejection_reason != nullptr) {
            *rejection_reason = "num_clients exceeds fabric mux channel count limit";
        }
        return false;
    }

    if (benchmark_case.num_buffers_per_channel == 0) {
        if (rejection_reason != nullptr) {
            *rejection_reason = "num_buffers_per_channel must be greater than 0";
        }
        return false;
    }

    if (benchmark_case.num_packets_per_sender == 0) {
        if (rejection_reason != nullptr) {
            *rejection_reason = "num_packets_per_sender must be greater than 0";
        }
        return false;
    }

    if (variant.topology != topology_) {
        if (rejection_reason != nullptr) {
            *rejection_reason = "benchmark context topology does not match requested variant";
        }
        return false;
    }

    if (devices_.size() < 2) {
        if (rejection_reason != nullptr) {
            *rejection_reason = "fabric saturation benchmark requires at least 2 devices";
        }
        return false;
    }

    return prepare_common_saturation_run(*this, benchmark_case, rejection_reason).has_value();
}

SaturationRunResult run_mux_saturation_once(
    const FabricMuxSaturationBenchmarkContext& context,
    const SaturationVariant& variant,
    const SaturationCase& benchmark_case) {
    switch (variant.implementation) {
        case SaturationImplementation::V1: return run_v1_saturation_once(context, benchmark_case);
        case SaturationImplementation::V2: return run_v2_saturation_once(context, benchmark_case);
    }
    return make_error("requested saturation implementation is not recognized");
}

}  // namespace tt::tt_fabric::bench
