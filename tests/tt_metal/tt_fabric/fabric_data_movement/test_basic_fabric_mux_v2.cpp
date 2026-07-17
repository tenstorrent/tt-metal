// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "fabric_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"

namespace tt::tt_fabric::fabric_router_tests::fabric_mux_v2_tests {

namespace {

constexpr char kSenderKernelSrc[] =
    "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_mux_v2_sender_client.cpp";
constexpr char kReceiverKernelSrc[] =
    "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_mux_v2_receiver.cpp";

constexpr uint32_t kTestResultsSizeBytes = 128;
constexpr uint32_t kSeedStride = 0x9E3779B9u;
constexpr uint32_t kShortPacketCount = 1'000;
constexpr uint32_t kMediumPacketCount = 10'000;
constexpr uint32_t kLongPacketCount = 100'000;
constexpr uint32_t kDefaultReturnCreditsPerPacket = 16;

using MeshDevicePtr = std::shared_ptr<tt::tt_metal::distributed::MeshDevice>;

constexpr std::array<tt::tt_fabric::RoutingDirection, 4> kRoutingDirections = {
    tt::tt_fabric::RoutingDirection::N,
    tt::tt_fabric::RoutingDirection::S,
    tt::tt_fabric::RoutingDirection::E,
    tt::tt_fabric::RoutingDirection::W,
};

enum class ChannelBufferSizeKind : uint8_t {
    ExactFitAligned,
    LargerAligned,
};

enum class StagingTestPattern : uint32_t {
    BasicSend = 0,
    ZeroPacket = 1,
    StageThenFlush = 2,
    OpportunisticFlush = 3,
    StageAndClose = 4,
    StageRingFull = 5,
    StageIdle = 6,
};

struct TestCaseConfig {
    const char* name = "";
    uint32_t num_senders = 1;
    uint32_t num_packets = 0;
    uint32_t packet_payload_size_bytes = 0;
    uint8_t num_buffers_per_channel = 1;
    tt::tt_metal::NOC forwarder_noc = tt::tt_metal::NOC::RISCV_0_default;
    ChannelBufferSizeKind channel_buffer_size_kind = ChannelBufferSizeKind::ExactFitAligned;
    bool eager_staging = false;
    bool use_stateful_lane = false;
    StagingTestPattern test_pattern = StagingTestPattern::BasicSend;
    uint32_t stage_count = 0;
    uint32_t idle_cycles = 0;
    uint8_t status_read_trid = 0xFF;  // 0xFF = kInvalidStatusReadTrid (blocking fallback)
    // When set: per-packet randomized payload size (shared PRNG with receiver) and
    // sender-only random inter-packet delay. Host byte-count checks are skipped.
    bool randomize_payload_size_and_delay = false;
};

struct SenderMemoryLayout {
    uint32_t test_results_address = 0;
    uint32_t credit_handshake_address = 0;
    uint32_t packet_header_buffer_address = 0;
    uint32_t payload_buffer_address = 0;
};

struct ReceiverMemoryLayout {
    uint32_t test_results_address = 0;
    uint32_t packet_header_buffer_address = 0;
    uint32_t receiver_slots_base_address = 0;
};

struct RemoteReceiverDevice {
    tt::tt_fabric::FabricNodeId fabric_node_id;
    tt::tt_fabric::FabricNodeId anchor_dst_fabric_node_id;
    MeshDevicePtr device;
    CoreCoord receiver_mux_logical_core;
    std::vector<CoreCoord> receiver_logical_cores;
    uint32_t linear_num_hops = 0;
};

struct RoutingSelection {
    tt::tt_fabric::FabricNodeId src_fabric_node_id;
    MeshDevicePtr sender_device;
    tt::tt_fabric::FabricNodeId sender_anchor_dst_fabric_node_id;
    CoreCoord sender_mux_logical_core;
    std::vector<CoreCoord> sender_logical_cores;
    std::vector<RemoteReceiverDevice> remote_devices;
};

struct ReceiverEndpoint {
    tt::tt_fabric::FabricNodeId fabric_node_id;
    MeshDevicePtr device;
    CoreCoord logical_core;
    uint32_t linear_num_hops = 0;
};

struct SenderReceiverAssignment {
    uint8_t sender_logical_channel_id = 0;
    CoreCoord sender_logical_core;
    ReceiverEndpoint receiver;
    uint32_t seed = 0;
};

struct SenderReceiverRuntimeContext {
    SenderReceiverAssignment assignment;
    SenderMemoryLayout sender_memory;
    ReceiverMemoryLayout receiver_memory;
};

struct TestRuntimeConfig {
    uint32_t packet_payload_size_bytes = 0;
    uint32_t sender_channel_buffer_size_bytes = 0;
    bool use_mesh_api = false;
};

struct MuxDeployment {
    MeshDevicePtr device;
    std::shared_ptr<tt::tt_metal::Program> program;
    CoreCoord mux_virtual_core;
    std::unique_ptr<tt::tt_fabric::FabricMuxV2Config> mux_config;
    uint8_t next_logical_channel_id = 0;
};

struct ReceiverDeviceContext {
    ReceiverMemoryLayout receiver_memory;
    MuxDeployment receiver_mux_deployment;
};

ChipId get_physical_device_id(const MeshDevicePtr& device) { return device->get_devices()[0]->id(); }

uint32_t align_up(uint32_t value, uint32_t alignment) { return ((value + alignment - 1) / alignment) * alignment; }

bool is_2d_fabric() { return tt::tt_fabric::get_fabric_topology() == tt::tt_fabric::Topology::Mesh; }

uint32_t get_receiver_slot_count() { return kDefaultReturnCreditsPerPacket; }

uint32_t make_time_seed() {
    const auto seed = static_cast<uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    return seed == 0 ? 0xA5A5A5A5u : seed;
}

uint32_t get_l1_alignment() {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    return static_cast<uint32_t>(hal.get_alignment(tt::tt_metal::HalMemType::L1));
}

size_t get_worker_l1_end_address() {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    return hal.get_dev_addr(tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::BASE) +
           hal.get_dev_size(tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::BASE);
}

uint32_t get_aligned_packet_header_size_bytes() {
    return align_up(static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_packet_header_size_bytes()), get_l1_alignment());
}

uint32_t get_sender_channel_buffer_size_bytes(const TestCaseConfig& test_case, uint32_t packet_payload_size_bytes) {
    const uint32_t alignment = get_l1_alignment();
    const uint32_t exact_fit_size =
        align_up(get_aligned_packet_header_size_bytes() + packet_payload_size_bytes, alignment);
    if (test_case.channel_buffer_size_kind == ChannelBufferSizeKind::ExactFitAligned) {
        return exact_fit_size;
    }
    return exact_fit_size + alignment;
}

uint32_t resolve_packet_payload_size_bytes(const TestCaseConfig& test_case) {
    if (test_case.packet_payload_size_bytes != 0) {
        return test_case.packet_payload_size_bytes;
    }
    return static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes());
}

TestRuntimeConfig build_test_runtime_config(const TestCaseConfig& test_case) {
    const auto packet_payload_size_bytes = resolve_packet_payload_size_bytes(test_case);
    return TestRuntimeConfig{
        .packet_payload_size_bytes = packet_payload_size_bytes,
        .sender_channel_buffer_size_bytes = get_sender_channel_buffer_size_bytes(test_case, packet_payload_size_bytes),
        .use_mesh_api = is_2d_fabric(),
    };
}

std::vector<CoreCoord> enumerate_worker_cores(const MeshDevicePtr& device) {
    const auto grid_size = device->compute_with_storage_grid_size();
    std::vector<CoreCoord> worker_cores;
    worker_cores.reserve(static_cast<size_t>(grid_size.x) * static_cast<size_t>(grid_size.y));
    for (std::size_t y = 0; y < grid_size.y; ++y) {
        for (std::size_t x = 0; x < grid_size.x; ++x) {
            worker_cores.push_back(CoreCoord{static_cast<std::size_t>(x), static_cast<std::size_t>(y)});
        }
    }
    return worker_cores;
}

class AlignedL1Cursor {
public:
    explicit AlignedL1Cursor(const MeshDevicePtr& device) :
        alignment_(get_l1_alignment()),
        current_(align_up(
            static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1)),
            alignment_)) {}

    uint32_t reserve(uint32_t size_bytes) {
        const auto address = current_;
        current_ = align_up(current_ + size_bytes, alignment_);
        return address;
    }

    bool fits_in_worker_l1() const { return current_ <= get_worker_l1_end_address(); }

private:
    uint32_t alignment_;
    uint32_t current_;
};

std::optional<SenderMemoryLayout> try_build_sender_memory_layout(
    const MeshDevicePtr& device, uint32_t packet_payload_size_bytes) {
    AlignedL1Cursor cursor(device);
    SenderMemoryLayout layout{};
    layout.test_results_address = cursor.reserve(kTestResultsSizeBytes);
    layout.credit_handshake_address = cursor.reserve(get_l1_alignment());
    layout.packet_header_buffer_address = cursor.reserve(get_aligned_packet_header_size_bytes());
    layout.payload_buffer_address = cursor.reserve(packet_payload_size_bytes);
    return cursor.fits_in_worker_l1() ? std::optional<SenderMemoryLayout>{layout} : std::nullopt;
}

std::optional<ReceiverMemoryLayout> try_build_receiver_memory_layout(
    const MeshDevicePtr& device, uint32_t packet_payload_size_bytes, uint32_t receiver_slot_count) {
    AlignedL1Cursor cursor(device);
    ReceiverMemoryLayout layout{};
    layout.test_results_address = cursor.reserve(kTestResultsSizeBytes);
    layout.packet_header_buffer_address = cursor.reserve(get_aligned_packet_header_size_bytes());
    layout.receiver_slots_base_address = cursor.reserve(receiver_slot_count * packet_payload_size_bytes);
    return cursor.fits_in_worker_l1() ? std::optional<ReceiverMemoryLayout>{layout} : std::nullopt;
}

uint32_t get_receiver_mux_channel_buffer_size_bytes() { return get_aligned_packet_header_size_bytes(); }

std::vector<RemoteReceiverDevice> collect_1d_remote_devices_in_direction(
    BaseFabricFixture& fixture,
    const tt::tt_fabric::FabricNodeId& src_fabric_node_id,
    tt::tt_fabric::RoutingDirection direction) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    std::vector<RemoteReceiverDevice> remote_devices;
    auto current_fabric_node_id = src_fabric_node_id;
    uint32_t num_hops = 0;

    while (true) {
        const auto neighbors = control_plane.get_intra_chip_neighbors(current_fabric_node_id, direction);
        if (neighbors.empty()) {
            break;
        }

        // 1D mux setup expects the anchor destination to be the immediate next hop.
        const auto anchor_dst_fabric_node_id = current_fabric_node_id;
        current_fabric_node_id = tt::tt_fabric::FabricNodeId(src_fabric_node_id.mesh_id, neighbors[0]);
        num_hops += 1;

        auto receiver_device =
            fixture.get_device(control_plane.get_physical_chip_id_from_fabric_node_id(current_fabric_node_id));
        auto worker_cores = enumerate_worker_cores(receiver_device);
        if (worker_cores.size() < 2) {
            continue;
        }

        const auto receiver_mux_logical_core = worker_cores.front();
        worker_cores.erase(worker_cores.begin());
        remote_devices.push_back(RemoteReceiverDevice{
            current_fabric_node_id,
            anchor_dst_fabric_node_id,
            receiver_device,
            receiver_mux_logical_core,
            std::move(worker_cores),
            num_hops,
        });
    }

    return remote_devices;
}

std::vector<RemoteReceiverDevice> collect_2d_remote_devices_in_direction(
    BaseFabricFixture& fixture,
    const MeshDevicePtr& sender_device,
    const tt::tt_fabric::FabricNodeId& src_fabric_node_id,
    tt::tt_fabric::RoutingDirection direction) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    std::vector<RemoteReceiverDevice> remote_devices;

    for (const auto& candidate_device : fixture.get_devices()) {
        if (get_physical_device_id(candidate_device) == get_physical_device_id(sender_device)) {
            continue;
        }

        const auto dst_fabric_node_id =
            control_plane.get_fabric_node_id_from_physical_chip_id(get_physical_device_id(candidate_device));
        const auto forwarding_direction =
            control_plane.get_forwarding_direction(src_fabric_node_id, dst_fabric_node_id);
        if (!forwarding_direction.has_value() || forwarding_direction.value() != direction) {
            continue;
        }

        auto worker_cores = enumerate_worker_cores(candidate_device);
        if (worker_cores.size() < 2) {
            continue;
        }

        const auto receiver_mux_logical_core = worker_cores.front();
        worker_cores.erase(worker_cores.begin());
        remote_devices.push_back(RemoteReceiverDevice{
            dst_fabric_node_id,
            src_fabric_node_id,
            candidate_device,
            receiver_mux_logical_core,
            std::move(worker_cores),
            0,
        });
    }

    std::sort(remote_devices.begin(), remote_devices.end(), [](const auto& lhs, const auto& rhs) {
        return std::make_tuple(static_cast<uint32_t>(*lhs.fabric_node_id.mesh_id), lhs.fabric_node_id.chip_id) <
               std::make_tuple(static_cast<uint32_t>(*rhs.fabric_node_id.mesh_id), rhs.fabric_node_id.chip_id);
    });

    return remote_devices;
}

std::vector<RemoteReceiverDevice> collect_remote_devices_in_direction(
    BaseFabricFixture& fixture,
    const MeshDevicePtr& sender_device,
    const tt::tt_fabric::FabricNodeId& src_fabric_node_id,
    tt::tt_fabric::RoutingDirection direction) {
    if (is_2d_fabric()) {
        return collect_2d_remote_devices_in_direction(fixture, sender_device, src_fabric_node_id, direction);
    }
    return collect_1d_remote_devices_in_direction(fixture, src_fabric_node_id, direction);
}

std::size_t count_total_receiver_cores(const std::vector<RemoteReceiverDevice>& remote_devices) {
    std::size_t total_receiver_cores = 0;
    for (const auto& remote_device : remote_devices) {
        total_receiver_cores += remote_device.receiver_logical_cores.size();
    }
    return total_receiver_cores;
}

std::optional<RoutingSelection> select_routing_selection(BaseFabricFixture& fixture, const TestCaseConfig& test_case) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    for (const auto& sender_device : fixture.get_devices()) {
        auto worker_cores = enumerate_worker_cores(sender_device);
        if (worker_cores.size() < test_case.num_senders + 1) {
            continue;
        }

        const auto sender_mux_logical_core = worker_cores.front();
        worker_cores.erase(worker_cores.begin());
        if (worker_cores.size() < test_case.num_senders) {
            continue;
        }

        const auto src_fabric_node_id =
            control_plane.get_fabric_node_id_from_physical_chip_id(get_physical_device_id(sender_device));
        for (const auto direction : kRoutingDirections) {
            const auto neighbors = control_plane.get_intra_chip_neighbors(src_fabric_node_id, direction);
            if (neighbors.empty()) {
                continue;
            }

            auto remote_devices =
                collect_remote_devices_in_direction(fixture, sender_device, src_fabric_node_id, direction);
            if (count_total_receiver_cores(remote_devices) < test_case.num_senders) {
                continue;
            }

            return RoutingSelection{
                src_fabric_node_id,
                sender_device,
                tt::tt_fabric::FabricNodeId(src_fabric_node_id.mesh_id, neighbors[0]),
                sender_mux_logical_core,
                std::move(worker_cores),
                std::move(remote_devices),
            };
        }
    }

    return std::nullopt;
}

std::optional<std::vector<SenderReceiverAssignment>> build_sender_receiver_assignments(
    const TestCaseConfig& test_case, const RoutingSelection& routing_selection, uint32_t run_seed) {
    if (routing_selection.sender_logical_cores.size() < test_case.num_senders ||
        routing_selection.remote_devices.empty()) {
        return std::nullopt;
    }

    std::vector<SenderReceiverAssignment> assignments;
    assignments.reserve(test_case.num_senders);
    std::vector<std::size_t> next_receiver_core_index(routing_selection.remote_devices.size(), 0);
    std::size_t next_remote_device_index = 0;

    for (uint32_t sender_idx = 0; sender_idx < test_case.num_senders; ++sender_idx) {
        std::size_t attempts = 0;
        while (attempts < routing_selection.remote_devices.size() &&
               next_receiver_core_index[next_remote_device_index] >=
                   routing_selection.remote_devices[next_remote_device_index].receiver_logical_cores.size()) {
            next_remote_device_index = (next_remote_device_index + 1) % routing_selection.remote_devices.size();
            attempts += 1;
        }
        if (attempts == routing_selection.remote_devices.size()) {
            return std::nullopt;
        }

        const auto& remote_device = routing_selection.remote_devices[next_remote_device_index];
        const auto receiver_logical_core =
            remote_device.receiver_logical_cores[next_receiver_core_index[next_remote_device_index]++];
        const uint32_t sender_seed = run_seed ^ (sender_idx * kSeedStride);
        assignments.push_back(SenderReceiverAssignment{
            static_cast<uint8_t>(sender_idx),
            routing_selection.sender_logical_cores[sender_idx],
            ReceiverEndpoint{
                remote_device.fabric_node_id,
                remote_device.device,
                receiver_logical_core,
                remote_device.linear_num_hops,
            },
            sender_seed,
        });

        next_remote_device_index = (next_remote_device_index + 1) % routing_selection.remote_devices.size();
    }

    return assignments;
}

std::unordered_map<ChipId, uint8_t> get_sender_count_by_receiver_device(
    const std::vector<SenderReceiverAssignment>& assignments) {
    std::unordered_map<ChipId, uint8_t> sender_count_by_receiver_device;
    for (const auto& assignment : assignments) {
        sender_count_by_receiver_device[get_physical_device_id(assignment.receiver.device)] += 1;
    }
    return sender_count_by_receiver_device;
}

tt::tt_metal::KernelHandle create_worker_kernel(
    tt::tt_metal::Program& program,
    const char* kernel_src,
    const CoreCoord& logical_core,
    std::vector<uint32_t> compile_args) {
    std::map<std::string, std::string> defines = {};
    if (is_2d_fabric()) {
        defines["FABRIC_2D"] = "";
    }

    return tt::tt_metal::CreateKernel(
        program,
        kernel_src,
        {logical_core},
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = std::move(compile_args),
            .defines = std::move(defines),
        });
}

std::optional<MuxDeployment> create_mux_deployment(
    const MeshDevicePtr& device,
    const tt::tt_fabric::FabricNodeId& src_fabric_node_id,
    const tt::tt_fabric::FabricNodeId& anchor_dst_fabric_node_id,
    const CoreCoord& mux_logical_core,
    uint8_t num_channels,
    uint8_t num_buffers_per_channel,
    uint32_t channel_buffer_size_bytes,
    const TestCaseConfig& test_case) {
    const auto link_indices = tt::tt_fabric::get_forwarding_link_indices(src_fabric_node_id, anchor_dst_fabric_node_id);
    if (link_indices.empty()) {
        return std::nullopt;
    }

    auto program = std::make_shared<tt::tt_metal::Program>(tt::tt_metal::CreateProgram());
    auto mux_config = std::make_unique<tt::tt_fabric::FabricMuxV2Config>(
        num_channels,
        num_buffers_per_channel,
        channel_buffer_size_bytes,
        device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1));

    tt::tt_fabric::add_fabric_mux_v2_to_program(
        *program,
        *mux_config,
        mux_logical_core,
        src_fabric_node_id,
        anchor_dst_fabric_node_id,
        link_indices.front(),
        test_case.forwarder_noc);

    return MuxDeployment{
        device,
        std::move(program),
        device->worker_core_from_logical_core(mux_logical_core),
        std::move(mux_config),
        0,
    };
}

std::optional<std::unordered_map<ChipId, ReceiverDeviceContext>> build_receiver_device_contexts(
    const RoutingSelection& routing_selection,
    const TestCaseConfig& test_case,
    const TestRuntimeConfig& test_runtime_config,
    const std::unordered_map<ChipId, uint8_t>& sender_count_by_receiver_device) {
    std::unordered_map<ChipId, ReceiverDeviceContext> receiver_device_contexts;
    receiver_device_contexts.reserve(sender_count_by_receiver_device.size());

    for (const auto& remote_device : routing_selection.remote_devices) {
        const auto receiver_device_id = get_physical_device_id(remote_device.device);
        auto count_it = sender_count_by_receiver_device.find(receiver_device_id);
        if (count_it == sender_count_by_receiver_device.end() || count_it->second == 0) {
            continue;
        }

        auto receiver_memory = try_build_receiver_memory_layout(
            remote_device.device, test_runtime_config.packet_payload_size_bytes, get_receiver_slot_count());
        if (!receiver_memory.has_value()) {
            return std::nullopt;
        }

        auto receiver_mux_deployment = create_mux_deployment(
            remote_device.device,
            remote_device.fabric_node_id,
            remote_device.anchor_dst_fabric_node_id,
            remote_device.receiver_mux_logical_core,
            count_it->second,
            test_case.num_buffers_per_channel,
            get_receiver_mux_channel_buffer_size_bytes(),
            test_case);
        if (!receiver_mux_deployment.has_value()) {
            return std::nullopt;
        }

        receiver_device_contexts.emplace(
            receiver_device_id,
            ReceiverDeviceContext{
                receiver_memory.value(),
                std::move(receiver_mux_deployment.value()),
            });
    }

    return receiver_device_contexts;
}

std::vector<uint32_t> make_common_compile_args(
    uint32_t test_results_address,
    const SenderMemoryLayout& sender_memory,
    const ReceiverMemoryLayout& receiver_memory,
    const TestRuntimeConfig& test_runtime_config,
    const TestCaseConfig& test_case) {
    return {
        test_results_address,
        kTestResultsSizeBytes,
        receiver_memory.receiver_slots_base_address,
        sender_memory.credit_handshake_address,
        test_runtime_config.use_mesh_api ? 1u : 0u,
        test_case.eager_staging ? 1u : 0u,
        test_case.use_stateful_lane ? 1u : 0u,
        static_cast<uint32_t>(test_case.test_pattern),
        static_cast<uint32_t>(test_case.status_read_trid),
        test_case.randomize_payload_size_and_delay ? 1u : 0u,
    };
}

void bind_worker_to_mux_channel(
    MuxDeployment& mux_deployment,
    const CoreCoord& worker_logical_core,
    tt::tt_metal::KernelHandle kernel,
    std::vector<uint32_t>& runtime_args) {
    const auto flow_control_sem_id = tt::tt_metal::CreateSemaphore(*mux_deployment.program, worker_logical_core, 0);
    const auto teardown_sem_id = tt::tt_metal::CreateSemaphore(*mux_deployment.program, worker_logical_core, 0);
    mux_deployment.mux_config->append_client_connection_rt_args(
        mux_deployment.mux_virtual_core,
        mux_deployment.next_logical_channel_id++,
        tt::tt_fabric::FabricMuxV2Config::ClientSemaphores{
            flow_control_sem_id,
            teardown_sem_id,
        },
        runtime_args);
    tt::tt_metal::SetRuntimeArgs(*mux_deployment.program, kernel, worker_logical_core, runtime_args);
}

std::vector<uint32_t> make_receiver_runtime_args(
    const TestCaseConfig& test_case,
    const TestRuntimeConfig& test_runtime_config,
    const SenderReceiverAssignment& assignment,
    const RoutingSelection& routing_selection,
    const ReceiverMemoryLayout& receiver_memory,
    const CoreCoord& sender_virtual_core) {
    return {
        receiver_memory.packet_header_buffer_address,
        test_runtime_config.packet_payload_size_bytes,
        test_case.num_packets,
        assignment.seed,
        kDefaultReturnCreditsPerPacket,
        static_cast<uint32_t>(sender_virtual_core.x),
        static_cast<uint32_t>(sender_virtual_core.y),
        get_receiver_slot_count(),
        test_runtime_config.use_mesh_api ? 0u : assignment.receiver.linear_num_hops,
        static_cast<uint32_t>(routing_selection.src_fabric_node_id.chip_id),
        static_cast<uint32_t>(*routing_selection.src_fabric_node_id.mesh_id),
    };
}

std::vector<uint32_t> make_sender_runtime_args(
    const TestCaseConfig& test_case,
    const TestRuntimeConfig& test_runtime_config,
    const SenderReceiverAssignment& assignment,
    const SenderMemoryLayout& sender_memory,
    const CoreCoord& receiver_virtual_core) {
    const uint32_t effective_stage_count = std::min(
        {test_case.stage_count, test_case.num_packets, static_cast<uint32_t>(test_case.num_buffers_per_channel)});

    return {
        sender_memory.packet_header_buffer_address,
        sender_memory.payload_buffer_address,
        test_runtime_config.packet_payload_size_bytes,
        test_case.num_packets,
        assignment.seed,
        static_cast<uint32_t>(receiver_virtual_core.x),
        static_cast<uint32_t>(receiver_virtual_core.y),
        get_receiver_slot_count(),
        test_runtime_config.use_mesh_api ? 0u : assignment.receiver.linear_num_hops,
        static_cast<uint32_t>(assignment.receiver.fabric_node_id.chip_id),
        static_cast<uint32_t>(*assignment.receiver.fabric_node_id.mesh_id),
        effective_stage_count,
        test_case.idle_cycles,
    };
}

uint64_t read_word_count(const std::vector<uint32_t>& worker_status) {
    return (static_cast<uint64_t>(worker_status[TT_FABRIC_WORD_CNT_INDEX + 1]) << 32) |
           worker_status[TT_FABRIC_WORD_CNT_INDEX];
}

std::vector<uint32_t> read_worker_status(
    const MeshDevicePtr& device, const CoreCoord& logical_core, uint32_t test_results_address) {
    std::vector<uint32_t> worker_status;
    tt::tt_metal::detail::ReadFromDeviceL1(
        device->get_devices()[0],
        logical_core,
        test_results_address,
        kTestResultsSizeBytes,
        worker_status,
        CoreType::WORKER);
    return worker_status;
}

void run_test_case(BaseFabricFixture& fixture, const TestCaseConfig& test_case) {
    ASSERT_GT(get_receiver_slot_count(), 0u);

    const auto test_runtime_config = build_test_runtime_config(test_case);
    ASSERT_GT(test_runtime_config.packet_payload_size_bytes, 0u);
    ASSERT_EQ(test_runtime_config.packet_payload_size_bytes % 16u, 0u);

    ASSERT_LE(
        test_runtime_config.sender_channel_buffer_size_bytes, tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes())
        << "Case " << test_case.name << " requested sender mux channel buffer size "
        << test_runtime_config.sender_channel_buffer_size_bytes << " but fabric limit is "
        << tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();

    auto routing_selection = select_routing_selection(fixture, test_case);
    if (!routing_selection.has_value()) {
        GTEST_SKIP() << "Case " << test_case.name
                     << " could not find enough worker cores and remote receivers for the mux-v2 setup requirements";
    }

    const auto run_seed = make_time_seed();
    auto sender_receiver_assignments =
        build_sender_receiver_assignments(test_case, routing_selection.value(), run_seed);
    ASSERT_TRUE(sender_receiver_assignments.has_value())
        << "Case " << test_case.name << " could not assign sender and receiver worker cores";

    auto sender_memory =
        try_build_sender_memory_layout(routing_selection->sender_device, test_runtime_config.packet_payload_size_bytes);
    ASSERT_TRUE(sender_memory.has_value())
        << "Case " << test_case.name << " sender worker memory layout exceeds worker L1 budget for payload size "
        << test_runtime_config.packet_payload_size_bytes;

    const auto sender_count_by_receiver_device =
        get_sender_count_by_receiver_device(sender_receiver_assignments.value());
    auto receiver_device_contexts = build_receiver_device_contexts(
        routing_selection.value(), test_case, test_runtime_config, sender_count_by_receiver_device);
    ASSERT_TRUE(receiver_device_contexts.has_value())
        << "Case " << test_case.name
        << " could not build receiver mux deployment or receiver memory layout for one or more receiver devices";

    auto sender_mux_deployment = create_mux_deployment(
        routing_selection->sender_device,
        routing_selection->src_fabric_node_id,
        routing_selection->sender_anchor_dst_fabric_node_id,
        routing_selection->sender_mux_logical_core,
        static_cast<uint8_t>(test_case.num_senders),
        test_case.num_buffers_per_channel,
        test_runtime_config.sender_channel_buffer_size_bytes,
        test_case);
    ASSERT_TRUE(sender_mux_deployment.has_value())
        << "Case " << test_case.name << " could not build sender mux deployment";

    std::vector<SenderReceiverRuntimeContext> sender_receiver_contexts;
    sender_receiver_contexts.reserve(sender_receiver_assignments->size());

    for (const auto& assignment : sender_receiver_assignments.value()) {
        auto& receiver_device_context =
            receiver_device_contexts->at(get_physical_device_id(assignment.receiver.device));
        auto& receiver_mux_deployment = receiver_device_context.receiver_mux_deployment;
        const auto& receiver_memory = receiver_device_context.receiver_memory;

        const auto sender_virtual_core =
            routing_selection->sender_device->worker_core_from_logical_core(assignment.sender_logical_core);
        const auto receiver_virtual_core =
            assignment.receiver.device->worker_core_from_logical_core(assignment.receiver.logical_core);

        const auto receiver_kernel = create_worker_kernel(
            *receiver_mux_deployment.program,
            kReceiverKernelSrc,
            assignment.receiver.logical_core,
            make_common_compile_args(
                receiver_memory.test_results_address,
                sender_memory.value(),
                receiver_memory,
                test_runtime_config,
                test_case));
        auto receiver_runtime_args = make_receiver_runtime_args(
            test_case,
            test_runtime_config,
            assignment,
            routing_selection.value(),
            receiver_memory,
            sender_virtual_core);
        bind_worker_to_mux_channel(
            receiver_mux_deployment, assignment.receiver.logical_core, receiver_kernel, receiver_runtime_args);

        const auto sender_kernel = create_worker_kernel(
            *sender_mux_deployment->program,
            kSenderKernelSrc,
            assignment.sender_logical_core,
            make_common_compile_args(
                sender_memory->test_results_address,
                sender_memory.value(),
                receiver_memory,
                test_runtime_config,
                test_case));
        auto sender_runtime_args = make_sender_runtime_args(
            test_case, test_runtime_config, assignment, sender_memory.value(), receiver_virtual_core);
        bind_worker_to_mux_channel(
            *sender_mux_deployment, assignment.sender_logical_core, sender_kernel, sender_runtime_args);

        sender_receiver_contexts.push_back(SenderReceiverRuntimeContext{
            assignment,
            sender_memory.value(),
            receiver_memory,
        });
    }

    for (const auto& receiver_device_context_entry : receiver_device_contexts.value()) {
        const auto& receiver_device_context = receiver_device_context_entry.second;
        fixture.RunProgramNonblocking(
            receiver_device_context.receiver_mux_deployment.device,
            *receiver_device_context.receiver_mux_deployment.program);
    }
    fixture.RunProgramNonblocking(sender_mux_deployment->device, *sender_mux_deployment->program);

    fixture.WaitForSingleProgramDone(sender_mux_deployment->device, *sender_mux_deployment->program);
    for (const auto& receiver_device_context_entry : receiver_device_contexts.value()) {
        const auto& receiver_device_context = receiver_device_context_entry.second;
        fixture.WaitForSingleProgramDone(
            receiver_device_context.receiver_mux_deployment.device,
            *receiver_device_context.receiver_mux_deployment.program);
    }

    const uint64_t expected_bytes =
        static_cast<uint64_t>(test_runtime_config.packet_payload_size_bytes) * test_case.num_packets;
    for (const auto& context : sender_receiver_contexts) {
        const auto sender_status = read_worker_status(
            sender_mux_deployment->device,
            context.assignment.sender_logical_core,
            context.sender_memory.test_results_address);
        const auto receiver_status = read_worker_status(
            context.assignment.receiver.device,
            context.assignment.receiver.logical_core,
            context.receiver_memory.test_results_address);

        EXPECT_EQ(sender_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS)
            << "Sender channel " << static_cast<uint32_t>(context.assignment.sender_logical_channel_id) << " failed";
        EXPECT_EQ(receiver_status[TT_FABRIC_STATUS_INDEX], TT_FABRIC_STATUS_PASS)
            << "Receiver channel " << static_cast<uint32_t>(context.assignment.sender_logical_channel_id) << " failed";

        if (!test_case.randomize_payload_size_and_delay) {
            EXPECT_EQ(read_word_count(sender_status), expected_bytes)
                << "Sender byte count mismatch for channel "
                << static_cast<uint32_t>(context.assignment.sender_logical_channel_id);
            EXPECT_EQ(read_word_count(receiver_status), expected_bytes)
                << "Receiver byte count mismatch for channel "
                << static_cast<uint32_t>(context.assignment.sender_logical_channel_id);
        }
    }
}

std::string test_case_name(const ::testing::TestParamInfo<TestCaseConfig>& info) { return info.param.name; }

}  // namespace

// Smoke: high-signal subset for N300 merge-gate / fast CI.
// Filter: --gtest_filter="*FabricMuxV2Smoke*Fixture.*"
class FabricMuxV2Smoke1DFixture : public Fabric1DFixture, public ::testing::WithParamInterface<TestCaseConfig> {};

TEST_P(FabricMuxV2Smoke1DFixture, SharedMuxFunctionalCoverage) { run_test_case(*this, GetParam()); }

class FabricMuxV2Smoke2DFixture : public Fabric2DFixture, public ::testing::WithParamInterface<TestCaseConfig> {};

TEST_P(FabricMuxV2Smoke2DFixture, SharedMuxFunctionalCoverage) { run_test_case(*this, GetParam()); }

// Full: remaining coverage for T3K / Galaxy / BH multi-card.
// Filter: --gtest_filter="*FabricMuxV2Functional*Fixture.*"
// Combined: --gtest_filter="*FabricMuxV2*Fixture.*"
class FabricMuxV2Functional1DFixture : public Fabric1DFixture, public ::testing::WithParamInterface<TestCaseConfig> {};

TEST_P(FabricMuxV2Functional1DFixture, SharedMuxFunctionalCoverage) { run_test_case(*this, GetParam()); }

class FabricMuxV2Functional2DFixture : public Fabric2DFixture, public ::testing::WithParamInterface<TestCaseConfig> {};

TEST_P(FabricMuxV2Functional2DFixture, SharedMuxFunctionalCoverage) { run_test_case(*this, GetParam()); }

constexpr std::array<TestCaseConfig, 6> kSmokeCases = {{
    TestCaseConfig{
        .name = "SingleSender_DefaultPayload_Riscv0",
        .num_packets = kShortPacketCount,
        .num_buffers_per_channel = 4,
    },
    TestCaseConfig{
        .name = "SingleSender_DefaultPayload_Riscv1",
        .num_packets = kShortPacketCount,
        .num_buffers_per_channel = 4,
        .forwarder_noc = tt::tt_metal::NOC::RISCV_1_default,
    },
    TestCaseConfig{
        .name = "MultiSender_16Senders",
        .num_senders = 16,
        .num_packets = kShortPacketCount,
        .packet_payload_size_bytes = 128,
        .num_buffers_per_channel = 1,
    },
    // Non-pow2 buffer counts exercise the forwarder generic slot-wrap path.
    TestCaseConfig{
        .name = "NonPow2Buffers_5Bufs_MultiSender",
        .num_senders = 8,
        .num_packets = kShortPacketCount,
        .packet_payload_size_bytes = 128,
        .num_buffers_per_channel = 5,
    },
    TestCaseConfig{
        .name = "Staging_NonStateful_StageRingFull",
        .num_packets = kShortPacketCount,
        .num_buffers_per_channel = 4,
        .eager_staging = true,
        .test_pattern = StagingTestPattern::StageRingFull,
        .stage_count = 4,
    },
    // Stateful lane + opportunistic flush + non-blocking status TRID.
    TestCaseConfig{
        .name = "Staging_Stateful_TridOppFlush",
        .num_packets = kShortPacketCount,
        .num_buffers_per_channel = 4,
        .eager_staging = true,
        .use_stateful_lane = true,
        .test_pattern = StagingTestPattern::OpportunisticFlush,
        .stage_count = 2,
        .status_read_trid = 1,
    },
}};

constexpr std::array<TestCaseConfig, 26> kFullCases = {{
    TestCaseConfig{
        .name = "SmallPayload_SteadyState",
        .num_packets = kMediumPacketCount,
        .packet_payload_size_bytes = 64,
        .num_buffers_per_channel = 4,
        .channel_buffer_size_kind = ChannelBufferSizeKind::LargerAligned,
    },
    TestCaseConfig{
        .name = "MultiSender_4Senders",
        .num_senders = 4,
        .num_packets = kMediumPacketCount,
        .packet_payload_size_bytes = 128,
        .num_buffers_per_channel = 4,
    },
    // Odd channel counts are not a separate codegen path, but cover stream-id /
    // scratch / manager scan sizing that pow2 sender counts miss.
    TestCaseConfig{
        .name = "OddChannels_3Senders",
        .num_senders = 3,
        .num_packets = kMediumPacketCount,
        .packet_payload_size_bytes = 128,
        .num_buffers_per_channel = 4,
    },
    TestCaseConfig{
        .name = "NonPow2Buffers_3Bufs_StageRingFull",
        .num_senders = 4,
        .num_packets = kShortPacketCount,
        .num_buffers_per_channel = 3,
        .eager_staging = true,
        .test_pattern = StagingTestPattern::StageRingFull,
        .stage_count = 3,
    },
    TestCaseConfig{
        .name = "Stress_MultiSender_DefaultPayload_Riscv0",
        .num_senders = 4,
        .num_packets = kLongPacketCount,
        .num_buffers_per_channel = 8,
    },
    TestCaseConfig{
        .name = "Stress_MultiSender_DefaultPayload_Riscv1",
        .num_senders = 4,
        .num_packets = kLongPacketCount,
        .num_buffers_per_channel = 8,
        .forwarder_noc = tt::tt_metal::NOC::RISCV_1_default,
    },
    // Eager staging tests — non-stateful lane
    TestCaseConfig{
        .name = "Staging_NonStateful_ZeroPacket",
        .num_packets = 0,
        .num_buffers_per_channel = 4,
        .eager_staging = true,
        .test_pattern = StagingTestPattern::ZeroPacket,
    },
    TestCaseConfig{
        .name = "Staging_NonStateful_StageThenFlush",
        .num_packets = kShortPacketCount,
        .num_buffers_per_channel = 4,
        .eager_staging = true,
        .test_pattern = StagingTestPattern::StageThenFlush,
        .stage_count = 2,
    },
    TestCaseConfig{
        .name = "Staging_NonStateful_OppFlush",
        .num_packets = kShortPacketCount,
        .num_buffers_per_channel = 4,
        .eager_staging = true,
        .test_pattern = StagingTestPattern::OpportunisticFlush,
        .stage_count = 2,
    },
    TestCaseConfig{
        .name = "Staging_NonStateful_StageAndClose",
        .num_packets = 1,
        .num_buffers_per_channel = 4,
        .eager_staging = true,
        .test_pattern = StagingTestPattern::StageAndClose,
        .stage_count = 1,
    },
    TestCaseConfig{
        .name = "Staging_NonStateful_StageIdle",
        .num_packets = kShortPacketCount,
        .num_buffers_per_channel = 4,
        .eager_staging = true,
        .test_pattern = StagingTestPattern::StageIdle,
        .stage_count = 1,
        .idle_cycles = 1000,
    },
    // Eager staging tests — stateful lane
    TestCaseConfig{
        .name = "Staging_Stateful_ZeroPacket",
        .num_packets = 0,
        .num_buffers_per_channel = 4,
        .eager_staging = true,
        .use_stateful_lane = true,
        .test_pattern = StagingTestPattern::ZeroPacket,
    },
    TestCaseConfig{
        .name = "Staging_Stateful_StageThenFlush",
        .num_packets = kShortPacketCount,
        .num_buffers_per_channel = 4,
        .eager_staging = true,
        .use_stateful_lane = true,
        .test_pattern = StagingTestPattern::StageThenFlush,
        .stage_count = 2,
    },
    TestCaseConfig{
        .name = "Staging_Stateful_StageAndClose",
        .num_packets = 1,
        .num_buffers_per_channel = 4,
        .eager_staging = true,
        .use_stateful_lane = true,
        .test_pattern = StagingTestPattern::StageAndClose,
        .stage_count = 1,
    },
    TestCaseConfig{
        .name = "Staging_Stateful_StageRingFull",
        .num_packets = kShortPacketCount,
        .num_buffers_per_channel = 4,
        .eager_staging = true,
        .use_stateful_lane = true,
        .test_pattern = StagingTestPattern::StageRingFull,
        .stage_count = 4,
    },
    TestCaseConfig{
        .name = "Staging_Stateful_StageIdle",
        .num_packets = kShortPacketCount,
        .num_buffers_per_channel = 4,
        .eager_staging = true,
        .use_stateful_lane = true,
        .test_pattern = StagingTestPattern::StageIdle,
        .stage_count = 1,
        .idle_cycles = 1000,
    },
    // Eager staging tests with TRID — exercises non-blocking status read path
    TestCaseConfig{
        .name = "Staging_NonStateful_TridOppFlush",
        .num_packets = kShortPacketCount,
        .num_buffers_per_channel = 4,
        .eager_staging = true,
        .test_pattern = StagingTestPattern::OpportunisticFlush,
        .stage_count = 2,
        .status_read_trid = 1,
    },
    TestCaseConfig{
        .name = "Staging_NonStateful_TridStageAndClose",
        .num_packets = 1,
        .num_buffers_per_channel = 4,
        .eager_staging = true,
        .test_pattern = StagingTestPattern::StageAndClose,
        .stage_count = 1,
        .status_read_trid = 1,
    },
    TestCaseConfig{
        .name = "Staging_Stateful_TridStageAndClose",
        .num_packets = 1,
        .num_buffers_per_channel = 4,
        .eager_staging = true,
        .use_stateful_lane = true,
        .test_pattern = StagingTestPattern::StageAndClose,
        .stage_count = 1,
        .status_read_trid = 1,
    },
    // Multi-sender staging stress tests
    TestCaseConfig{
        .name = "Staging_Stress_8Senders_4Bufs_StageThenFlush",
        .num_senders = 8,
        .num_packets = kShortPacketCount,
        .num_buffers_per_channel = 4,
        .eager_staging = true,
        .test_pattern = StagingTestPattern::StageThenFlush,
        .stage_count = 2,
    },
    TestCaseConfig{
        .name = "Staging_Stress_8Senders_16Bufs_DeepStage",
        .num_senders = 8,
        .num_packets = kShortPacketCount,
        .num_buffers_per_channel = 16,
        .eager_staging = true,
        .test_pattern = StagingTestPattern::StageRingFull,
        .stage_count = 16,
    },
    TestCaseConfig{
        .name = "Staging_Stress_16Senders_1Buf_RingFullOnFirst",
        .num_senders = 16,
        .num_packets = kShortPacketCount,
        .num_buffers_per_channel = 1,
        .eager_staging = true,
        .test_pattern = StagingTestPattern::StageRingFull,
        .stage_count = 1,
    },
    // High channel count stress test
    TestCaseConfig{
        .name = "Stress_48Senders_DefaultPayload",
        .num_senders = 48,
        .num_packets = kLongPacketCount,
        .num_buffers_per_channel = 4,
    },
    TestCaseConfig{
        .name = "Stress_64Senders_DefaultPayload",
        .num_senders = 64,
        .num_packets = kLongPacketCount,
        .num_buffers_per_channel = 4,
    },
    // High channel count stress with per-packet randomized size + sender-only delay.
    // Stresses forwarder header-size freshness under slot reuse; host byte-count checks skipped.
    TestCaseConfig{
        .name = "Stress_48Senders_RandomSizeDelay",
        .num_senders = 48,
        .num_packets = kLongPacketCount,
        .num_buffers_per_channel = 4,
        .randomize_payload_size_and_delay = true,
    },
    TestCaseConfig{
        .name = "Stress_64Senders_RandomSizeDelay",
        .num_senders = 64,
        .num_packets = kLongPacketCount,
        .num_buffers_per_channel = 4,
        .randomize_payload_size_and_delay = true,
    },
}};

INSTANTIATE_TEST_SUITE_P(Smoke1D, FabricMuxV2Smoke1DFixture, ::testing::ValuesIn(kSmokeCases), test_case_name);
INSTANTIATE_TEST_SUITE_P(Smoke2D, FabricMuxV2Smoke2DFixture, ::testing::ValuesIn(kSmokeCases), test_case_name);

INSTANTIATE_TEST_SUITE_P(Functional1D, FabricMuxV2Functional1DFixture, ::testing::ValuesIn(kFullCases), test_case_name);
INSTANTIATE_TEST_SUITE_P(Functional2D, FabricMuxV2Functional2DFixture, ::testing::ValuesIn(kFullCases), test_case_name);

}  // namespace tt::tt_fabric::fabric_router_tests::fabric_mux_v2_tests
