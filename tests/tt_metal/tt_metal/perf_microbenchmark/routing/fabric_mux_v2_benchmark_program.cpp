#include "fabric_mux_v2_benchmark_program.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"
#include "fabric_mux_benchmark_program_utils.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_interface.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"

namespace tt::tt_fabric::bench {

namespace {

constexpr char kStandaloneSenderKernelPath[] =
    "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_mux_v2_ubench_sender.cpp";
constexpr char kStandaloneDrainerKernelPath[] =
    "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_mux_v2_ubench_drainer.cpp";

constexpr uint32_t kTestResultsSizeBytes = 128;
constexpr uint32_t kPacketHeaderBufferSizeBytes = 1024;
constexpr uint32_t kNocAddressPaddingBytes = 16;
constexpr uint32_t kStandaloneDrainerNumBuffers = 16;
constexpr uint32_t kSenderStartDelayCycles = 100000;
constexpr uint32_t kStandaloneDownstreamFreeSlotsStreamId =
    tt::tt_fabric::connection_interface::sender_channel_0_free_slots_stream_id;

uint32_t low_u32(uint64_t value) { return static_cast<uint32_t>(value & 0xFFFFFFFFull); }

uint32_t high_u32(uint64_t value) { return static_cast<uint32_t>(value >> 32); }

class StandaloneMuxV2DrainerLayout {
public:
    StandaloneMuxV2DrainerLayout(uint8_t num_buffers, size_t buffer_size_bytes, size_t base_l1_address) :
        num_buffers_(num_buffers), buffer_size_bytes_(buffer_size_bytes) {
        TT_FATAL(num_buffers_ > 0, "StandaloneMuxV2DrainerLayout requires at least one buffer");
        TT_FATAL(buffer_size_bytes_ > 0, "StandaloneMuxV2DrainerLayout requires a non-zero buffer size");
        TT_FATAL(
            buffer_size_bytes_ <= tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes(),
            "StandaloneMuxV2DrainerLayout buffer size must be <= {}, got {}",
            tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes(),
            buffer_size_bytes_);

        const auto& hal = tt::tt_metal::MetalContext::instance().hal();
        noc_aligned_address_size_bytes_ = hal.get_alignment(tt::tt_metal::HalMemType::L1);

        size_t current_address = align_up(
            to_uint32_checked(base_l1_address, "drainer_layout_base_l1_address"), noc_aligned_address_size_bytes_);

        status_region_ = MemoryRegion(current_address, noc_aligned_address_size_bytes_);
        current_address = status_region_.get_end_address();

        current_address = align_up(current_address, noc_aligned_address_size_bytes_);
        connection_info_region_ = MemoryRegion(current_address, sizeof(tt::tt_fabric::EDMChannelWorkerLocationInfo));
        current_address = connection_info_region_.get_end_address();

        current_address = align_up(current_address, noc_aligned_address_size_bytes_);
        connection_handshake_region_ = MemoryRegion(current_address, noc_aligned_address_size_bytes_);
        current_address = connection_handshake_region_.get_end_address();

        current_address = align_up(current_address, noc_aligned_address_size_bytes_);
        buffer_index_region_ = MemoryRegion(current_address, noc_aligned_address_size_bytes_);
        current_address = buffer_index_region_.get_end_address();

        current_address = align_up(current_address, noc_aligned_address_size_bytes_);
        channel_region_ = MemoryRegion(current_address, static_cast<size_t>(num_buffers_) * buffer_size_bytes_);
        current_address = channel_region_.get_end_address();

        memory_map_end_address_ = current_address;
        TT_FATAL(
            memory_map_end_address_ <= get_worker_l1_end_address(),
            "StandaloneMuxV2DrainerLayout end address {} exceeds worker L1 end {}",
            memory_map_end_address_,
            get_worker_l1_end_address());
    }

    uint8_t get_num_buffers() const { return num_buffers_; }
    size_t get_buffer_size_bytes() const { return buffer_size_bytes_; }
    size_t get_status_address() const { return status_region_.get_address(); }
    size_t get_connection_info_address() const { return connection_info_region_.get_address(); }
    size_t get_connection_handshake_address() const { return connection_handshake_region_.get_address(); }
    size_t get_buffer_index_address() const { return buffer_index_region_.get_address(); }
    size_t get_channel_base_address() const { return channel_region_.get_address(); }
    size_t get_memory_map_end_address() const { return memory_map_end_address_; }

private:
    struct MemoryRegion {
        size_t base_address = 0;
        size_t size_bytes = 0;

        MemoryRegion() = default;
        MemoryRegion(size_t base, size_t size) : base_address(base), size_bytes(size) {}

        size_t get_address() const { return base_address; }
        size_t get_end_address() const { return base_address + size_bytes; }
        size_t get_total_size() const { return size_bytes; }
    };

    size_t noc_aligned_address_size_bytes_ = 0;
    uint8_t num_buffers_ = 0;
    size_t buffer_size_bytes_ = 0;
    MemoryRegion status_region_{};
    MemoryRegion connection_info_region_{};
    MemoryRegion connection_handshake_region_{};
    MemoryRegion buffer_index_region_{};
    MemoryRegion channel_region_{};
    size_t memory_map_end_address_ = 0;
};

struct SenderMemoryMap {
    uint32_t test_results_address = 0;
    uint32_t local_poll_scratch_address = 0;
    uint32_t start_signal_address = 0;
    uint32_t ready_count_address = 0;
    uint32_t packet_header_buffer_address = 0;
    uint32_t payload_buffer_address = 0;
    uint32_t dummy_target_address = 0;
    uint32_t end_address = 0;
};

SenderMemoryMap create_sender_memory_map(uint32_t base_l1_address, uint32_t payload_size_bytes) {
    SenderMemoryMap memory_map{};
    uint32_t current_address = base_l1_address;

    memory_map.test_results_address = current_address;
    current_address += kTestResultsSizeBytes;

    current_address = align_up(current_address, kNocAddressPaddingBytes);
    memory_map.local_poll_scratch_address = current_address;
    current_address += kNocAddressPaddingBytes;

    current_address = align_up(current_address, kNocAddressPaddingBytes);
    memory_map.start_signal_address = current_address;
    current_address += kNocAddressPaddingBytes;

    current_address = align_up(current_address, kNocAddressPaddingBytes);
    memory_map.ready_count_address = current_address;
    current_address += kNocAddressPaddingBytes;

    current_address = align_up(current_address, kNocAddressPaddingBytes);
    memory_map.packet_header_buffer_address = current_address;
    current_address += kPacketHeaderBufferSizeBytes;

    current_address = align_up(current_address, kNocAddressPaddingBytes);
    memory_map.payload_buffer_address = current_address;
    memory_map.dummy_target_address = current_address;
    current_address += payload_size_bytes;

    memory_map.end_address = current_address;
    return memory_map;
}

void validate_sender_memory_map(const SenderMemoryMap& memory_map) {
    TT_FATAL(
        memory_map.end_address <= get_worker_l1_end_address(),
        "Standalone mux-v2 sender memory map end address {} exceeds worker L1 end {}",
        memory_map.end_address,
        get_worker_l1_end_address());
}

void initialize_drainer_state(
    tt::tt_metal::IDevice* device,
    const CoreCoord& drainer_logical_core,
    const StandaloneMuxV2DrainerLayout& drainer_layout) {
    write_zero_words_to_device(device, drainer_logical_core, drainer_layout.get_status_address(), sizeof(uint32_t));
    write_zero_words_to_device(
        device,
        drainer_logical_core,
        drainer_layout.get_connection_info_address(),
        sizeof(tt::tt_fabric::EDMChannelWorkerLocationInfo));
    write_zero_words_to_device(
        device, drainer_logical_core, drainer_layout.get_connection_handshake_address(), sizeof(uint32_t));
    write_zero_words_to_device(
        device, drainer_logical_core, drainer_layout.get_buffer_index_address(), sizeof(uint32_t));
}

std::vector<uint32_t> build_mux_downstream_sender_rt_args(
    tt::tt_metal::IDevice* device,
    tt::tt_metal::Program& program,
    const CoreCoord& mux_logical_core,
    const CoreCoord& drainer_virtual_core,
    const StandaloneMuxV2DrainerLayout& drainer_layout) {
    const auto worker_teardown_semaphore_id = tt::tt_metal::CreateSemaphore(program, mux_logical_core, 0);
    const auto worker_buffer_index_semaphore_id = tt::tt_metal::CreateSemaphore(program, mux_logical_core, 0);

    tt::tt_fabric::SenderWorkerAdapterSpec sender_worker_adapter_spec{
        .edm_noc_x = drainer_virtual_core.x,
        .edm_noc_y = drainer_virtual_core.y,
        .edm_buffer_base_addr = drainer_layout.get_channel_base_address(),
        .num_buffers_per_channel = drainer_layout.get_num_buffers(),
        .edm_l1_sem_addr = 0,
        .edm_connection_handshake_addr = drainer_layout.get_connection_handshake_address(),
        .edm_worker_location_info_addr = drainer_layout.get_connection_info_address(),
        .buffer_size_bytes = drainer_layout.get_buffer_size_bytes(),
        .buffer_index_semaphore_id = drainer_layout.get_buffer_index_address(),
        .edm_direction = tt::tt_fabric::eth_chan_directions::EAST,
    };

    std::vector<uint32_t> downstream_rt_args;
    tt::tt_fabric::append_worker_to_fabric_edm_sender_rt_args(
        sender_worker_adapter_spec,
        device->id(),
        {mux_logical_core},
        worker_teardown_semaphore_id,
        worker_buffer_index_semaphore_id,
        downstream_rt_args);
    return downstream_rt_args;
}

void create_drainer_kernel(
    tt::tt_metal::Program& program,
    const CoreCoord& drainer_logical_core,
    uint64_t expected_total_packets,
    const StandaloneMuxV2DrainerLayout& drainer_layout) {
    const std::vector<uint32_t> compile_args = {
        static_cast<uint32_t>(drainer_layout.get_num_buffers()),
        to_uint32_checked(drainer_layout.get_status_address(), "drainer_status_address"),
        to_uint32_checked(drainer_layout.get_connection_info_address(), "drainer_connection_info_address"),
        to_uint32_checked(drainer_layout.get_connection_handshake_address(), "drainer_connection_handshake_address"),
        kStandaloneDownstreamFreeSlotsStreamId,
    };

    const std::vector<uint32_t> runtime_args = {
        low_u32(expected_total_packets),
        high_u32(expected_total_packets),
    };

    create_data_movement_kernel(
        program, kStandaloneDrainerKernelPath, drainer_logical_core, compile_args, runtime_args);
}

void create_sender_kernel(
    tt::tt_metal::Program& program,
    const CoreCoord& sender_logical_core,
    const CoreCoord& drainer_virtual_core,
    const SenderMemoryMap& sender_memory_map,
    const tt::tt_fabric::FabricMuxV2Config& mux_config,
    const CoreCoord& mux_virtual_core,
    const StandaloneMuxV2DrainerLayout& drainer_layout,
    uint32_t logical_channel_id,
    uint32_t num_packets,
    uint32_t packet_payload_size_bytes,
    uint32_t dummy_receiver_noc_xy_encoding,
    uint32_t seed,
    uint32_t master_sender_noc_xy_encoding,
    uint32_t expected_ready_count,
    bool is_master_sender,
    const std::vector<uint32_t>& peer_sender_noc_xy_encodings) {
    const auto flow_control_sem_id = tt::tt_metal::CreateSemaphore(program, sender_logical_core, 0);
    const auto teardown_sem_id = tt::tt_metal::CreateSemaphore(program, sender_logical_core, 0);

    const std::vector<uint32_t> compile_args = {
        sender_memory_map.test_results_address,
        kTestResultsSizeBytes,
        sender_memory_map.start_signal_address,
        sender_memory_map.ready_count_address,
        sender_memory_map.local_poll_scratch_address,
        static_cast<uint32_t>(is_master_sender),
    };

    std::vector<uint32_t> runtime_args = {
        num_packets,
        packet_payload_size_bytes,
        sender_memory_map.packet_header_buffer_address,
        sender_memory_map.payload_buffer_address,
        sender_memory_map.dummy_target_address,
        dummy_receiver_noc_xy_encoding,
        seed,
        kSenderStartDelayCycles,
        static_cast<uint32_t>(drainer_virtual_core.x),
        static_cast<uint32_t>(drainer_virtual_core.y),
        to_uint32_checked(drainer_layout.get_status_address(), "drainer_status_address"),
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

    create_data_movement_kernel(program, kStandaloneSenderKernelPath, sender_logical_core, compile_args, runtime_args);
}

StandaloneMuxV2BenchmarkRunResult make_error(std::string error_message) {
    StandaloneMuxV2BenchmarkRunResult result{};
    result.error_message = std::move(error_message);
    return result;
}

StandaloneMuxV2BenchmarkRunResult to_run_result(const SenderResultSummary& summary) {
    StandaloneMuxV2BenchmarkRunResult result{};
    result.success = summary.success;
    result.error_message = summary.error_message;
    result.aggregate_bytes = summary.aggregate_bytes;
    result.max_sender_cycles = summary.max_sender_cycles;
    return result;
}

}  // namespace

StandaloneMuxV2BenchmarkRunResult run_standalone_mux_v2_benchmark_once(
    const FabricMuxV2BenchmarkContext& context, const MuxV2ThroughputCase& benchmark_case) {
    auto* device = context.get_device();
    const auto& mesh_device = context.get_mesh_device();
    TT_FATAL(device != nullptr, "Standalone mux-v2 benchmark requires an initialized device");
    TT_FATAL(mesh_device != nullptr, "Standalone mux-v2 benchmark requires an initialized mesh device");

    const auto resolved_payload_size_bytes = resolve_packet_payload_size_bytes(benchmark_case);
    const auto num_packets = derive_num_packets(benchmark_case);
    const uint64_t expected_total_packets = static_cast<uint64_t>(benchmark_case.num_senders) * num_packets;
    const uint64_t expected_sender_bytes = static_cast<uint64_t>(resolved_payload_size_bytes) * num_packets;
    const uint64_t expected_aggregate_bytes = expected_sender_bytes * benchmark_case.num_senders;

    const auto l1_unreserved_base_address =
        static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1));
    const auto packet_header_size_bytes =
        static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_packet_header_size_bytes());
    const auto channel_buffer_size_bytes = packet_header_size_bytes + resolved_payload_size_bytes;

    tt::tt_fabric::FabricMuxV2Config mux_config(
        static_cast<uint8_t>(benchmark_case.num_senders),
        benchmark_case.num_buffers_per_channel,
        channel_buffer_size_bytes,
        l1_unreserved_base_address);
    mux_config.set_forwarder_service_burst_size(benchmark_case.service_burst_size);
    mux_config.set_forwarder_max_in_flight_trids(benchmark_case.max_in_flight_trids);

    StandaloneMuxV2DrainerLayout drainer_layout(
        static_cast<uint8_t>(kStandaloneDrainerNumBuffers), channel_buffer_size_bytes, l1_unreserved_base_address);

    const auto mux_logical_core = context.get_mux_logical_core();
    const auto drainer_logical_core = context.get_drainer_logical_core();
    const auto sender_logical_cores = context.get_sender_logical_cores(benchmark_case);
    const auto mux_virtual_core = mesh_device->worker_core_from_logical_core(mux_logical_core);
    const auto drainer_virtual_core = mesh_device->worker_core_from_logical_core(drainer_logical_core);

    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const auto dummy_receiver_noc_xy_encoding =
        static_cast<uint32_t>(hal.noc_xy_encoding(drainer_virtual_core.x, drainer_virtual_core.y));

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    const auto mux_downstream_rt_args =
        build_mux_downstream_sender_rt_args(device, program, mux_logical_core, drainer_virtual_core, drainer_layout);
    tt::tt_fabric::add_fabric_mux_v2_to_program(
        program, mux_config, mux_logical_core, mux_downstream_rt_args, benchmark_case.forwarder_noc);

    create_drainer_kernel(program, drainer_logical_core, expected_total_packets, drainer_layout);

    std::vector<SenderMemoryMap> sender_memory_maps;
    sender_memory_maps.reserve(sender_logical_cores.size());
    std::vector<SenderExecutionContext> sender_execution_contexts;
    sender_execution_contexts.reserve(sender_logical_cores.size());
    const auto sender_noc_xy_encodings = build_sender_noc_xy_encodings(mesh_device, sender_logical_cores);
    TT_FATAL(!sender_noc_xy_encodings.empty(), "Standalone mux-v2 benchmark requires at least one sender");
    const auto master_sender_noc_xy_encoding = sender_noc_xy_encodings.front();
    const auto expected_ready_count = to_uint32_checked(sender_logical_cores.size(), "expected_ready_count");

    for (std::size_t sender_idx = 0; sender_idx < sender_logical_cores.size(); ++sender_idx) {
        const auto memory_map = create_sender_memory_map(l1_unreserved_base_address, resolved_payload_size_bytes);
        validate_sender_memory_map(memory_map);
        sender_memory_maps.push_back(memory_map);
        sender_execution_contexts.push_back(SenderExecutionContext{
            .logical_core = sender_logical_cores[sender_idx],
            .test_results_address = sender_memory_maps.back().test_results_address,
        });
        initialize_sender_start_barrier_state(
            device,
            sender_logical_cores[sender_idx],
            sender_memory_maps.back().start_signal_address,
            sender_memory_maps.back().ready_count_address,
            sender_idx == 0 ? 1u : 0u);

        auto peer_sender_noc_xy_encodings = build_peer_sender_noc_xy_encodings(sender_noc_xy_encodings, sender_idx);

        create_sender_kernel(
            program,
            sender_logical_cores[sender_idx],
            drainer_virtual_core,
            sender_memory_maps.back(),
            mux_config,
            mux_virtual_core,
            drainer_layout,
            static_cast<uint32_t>(sender_idx),
            num_packets,
            resolved_payload_size_bytes,
            dummy_receiver_noc_xy_encoding,
            0x12340000u + static_cast<uint32_t>(sender_idx),
            master_sender_noc_xy_encoding,
            expected_ready_count,
            sender_idx == 0,
            peer_sender_noc_xy_encodings);
    }

    initialize_drainer_state(device, drainer_logical_core, drainer_layout);
    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(device->id());
    enqueue_single_device_mesh_program(mesh_device, device->id(), std::move(program));

    std::vector<uint32_t> drainer_status;
    tt::tt_metal::detail::ReadFromDeviceL1(
        device, drainer_logical_core, drainer_layout.get_status_address(), sizeof(uint32_t), drainer_status);
    if (drainer_status.empty() || drainer_status[0] != tt::tt_fabric::EDMStatus::TERMINATED) {
        std::ostringstream message;
        message << "drainer did not terminate cleanly; status=" << (drainer_status.empty() ? 0u : drainer_status[0]);
        return make_error(message.str());
    }

    return to_run_result(read_and_validate_sender_results(
        device, sender_execution_contexts, kTestResultsSizeBytes, expected_sender_bytes, expected_aggregate_bytes));
}

}  // namespace tt::tt_fabric::bench
