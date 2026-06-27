#include "fabric_mux_benchmark_program_utils.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "context/metal_context.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"

namespace tt::tt_fabric::bench {

namespace {

SenderResultSummary make_error(std::string error_message) {
    SenderResultSummary result{};
    result.error_message = std::move(error_message);
    return result;
}

}  // namespace

uint32_t align_up(uint32_t value, uint32_t alignment) {
    TT_FATAL(alignment != 0, "Alignment must be non-zero");
    const uint32_t remainder = value % alignment;
    return remainder == 0 ? value : (value + (alignment - remainder));
}

uint32_t to_uint32_checked(size_t value, const char* field_name) {
    TT_FATAL(value <= std::numeric_limits<uint32_t>::max(), "{} exceeds uint32_t range: {}", field_name, value);
    return static_cast<uint32_t>(value);
}

uint64_t combine_u64(uint32_t low, uint32_t high) {
    return static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32);
}

uint32_t get_worker_l1_end_address() {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    return static_cast<uint32_t>(
        hal.get_dev_addr(tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::BASE) +
        hal.get_dev_size(tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::BASE));
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

void write_zero_words_to_device(
    tt::tt_metal::IDevice* device, const CoreCoord& logical_core, size_t address, size_t size_bytes) {
    TT_FATAL(
        (size_bytes % sizeof(uint32_t)) == 0,
        "Zero-init region size {} must be a multiple of {}",
        size_bytes,
        sizeof(uint32_t));
    std::vector<uint32_t> zero_words(size_bytes / sizeof(uint32_t), 0);
    tt::tt_metal::detail::WriteToDeviceL1(
        device, logical_core, to_uint32_checked(address, "zero_init_address"), zero_words);
}

void write_word_to_device(
    tt::tt_metal::IDevice* device, const CoreCoord& logical_core, size_t address, uint32_t value) {
    std::vector<uint32_t> word_buffer{value};
    tt::tt_metal::detail::WriteToDeviceL1(
        device, logical_core, to_uint32_checked(address, "word_init_address"), word_buffer);
}

void initialize_sender_start_barrier_state(
    tt::tt_metal::IDevice* device,
    const CoreCoord& sender_logical_core,
    uint32_t start_signal_address,
    uint32_t ready_count_address,
    uint32_t initial_ready_count) {
    write_word_to_device(device, sender_logical_core, start_signal_address, 0);
    write_word_to_device(device, sender_logical_core, ready_count_address, initial_ready_count);
}

std::vector<uint32_t> build_sender_noc_xy_encodings(
    const MeshDevicePtr& sender_device, const std::vector<CoreCoord>& sender_logical_cores) {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    std::vector<uint32_t> sender_noc_xy_encodings;
    sender_noc_xy_encodings.reserve(sender_logical_cores.size());
    for (const auto& sender_logical_core : sender_logical_cores) {
        const auto sender_virtual_core = sender_device->worker_core_from_logical_core(sender_logical_core);
        sender_noc_xy_encodings.push_back(
            static_cast<uint32_t>(hal.noc_xy_encoding(sender_virtual_core.x, sender_virtual_core.y)));
    }
    return sender_noc_xy_encodings;
}

std::vector<uint32_t> build_peer_sender_noc_xy_encodings(
    const std::vector<uint32_t>& sender_noc_xy_encodings, std::size_t sender_idx) {
    if (sender_idx != 0 || sender_noc_xy_encodings.size() <= 1) {
        return {};
    }

    std::vector<uint32_t> peer_sender_noc_xy_encodings;
    peer_sender_noc_xy_encodings.reserve(sender_noc_xy_encodings.size() - 1);
    for (std::size_t peer_idx = 1; peer_idx < sender_noc_xy_encodings.size(); ++peer_idx) {
        peer_sender_noc_xy_encodings.push_back(sender_noc_xy_encodings[peer_idx]);
    }
    return peer_sender_noc_xy_encodings;
}

void create_data_movement_kernel(
    tt::tt_metal::Program& program,
    const std::string& kernel_path,
    const CoreCoord& logical_core,
    const std::vector<uint32_t>& compile_args,
    const std::vector<uint32_t>& runtime_args,
    tt::tt_metal::NOC noc) {
    const auto kernel_handle = tt::tt_metal::CreateKernel(
        program,
        kernel_path,
        {logical_core},
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = noc,
            .compile_args = compile_args,
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});
    tt::tt_metal::SetRuntimeArgs(program, kernel_handle, logical_core, runtime_args);
}

void enqueue_single_device_mesh_program(
    const MeshDevicePtr& mesh_device, ChipId physical_device_id, tt::tt_metal::Program&& program) {
    const auto target_coordinate = mesh_device->get_view().find_device(physical_device_id);
    const auto device_range = tt::tt_metal::distributed::MeshCoordinateRange(target_coordinate, target_coordinate);
    tt::tt_metal::distributed::MeshWorkload workload;
    workload.add_program(device_range, std::move(program));

    auto& command_queue = mesh_device->mesh_command_queue();
    tt::tt_metal::distributed::EnqueueMeshWorkload(command_queue, workload, false);
    tt::tt_metal::distributed::Finish(command_queue);
}

void enqueue_single_device_mesh_program(const MeshDevicePtr& mesh_device, tt::tt_metal::Program&& program) {
    const auto zero_coordinate =
        tt::tt_metal::distributed::MeshCoordinate::zero_coordinate(mesh_device->shape().dims());
    const auto device_range = tt::tt_metal::distributed::MeshCoordinateRange(zero_coordinate, zero_coordinate);
    tt::tt_metal::distributed::MeshWorkload workload;
    workload.add_program(device_range, std::move(program));

    auto& command_queue = mesh_device->mesh_command_queue();
    tt::tt_metal::distributed::EnqueueMeshWorkload(command_queue, workload, false);
    tt::tt_metal::distributed::Finish(command_queue);
}

SenderResultSummary read_and_validate_sender_results(
    tt::tt_metal::IDevice* device,
    const std::vector<SenderExecutionContext>& sender_execution_contexts,
    uint32_t test_results_size_bytes,
    uint64_t expected_sender_bytes,
    uint64_t expected_aggregate_bytes) {
    uint64_t aggregate_bytes = 0;
    uint64_t max_sender_cycles = 0;
    for (std::size_t sender_idx = 0; sender_idx < sender_execution_contexts.size(); ++sender_idx) {
        std::vector<uint32_t> sender_results;
        tt::tt_metal::detail::ReadFromDeviceL1(
            device,
            sender_execution_contexts[sender_idx].logical_core,
            sender_execution_contexts[sender_idx].test_results_address,
            test_results_size_bytes,
            sender_results);

        if (sender_results.size() < (test_results_size_bytes / sizeof(uint32_t))) {
            std::ostringstream message;
            message << "sender " << sender_idx << " returned truncated results";
            return make_error(message.str());
        }

        if (sender_results[TT_FABRIC_STATUS_INDEX] != TT_FABRIC_STATUS_PASS) {
            std::ostringstream message;
            message << "sender " << sender_idx << " failed with status " << sender_results[TT_FABRIC_STATUS_INDEX];
            return make_error(message.str());
        }

        const auto sender_bytes =
            combine_u64(sender_results[TT_FABRIC_WORD_CNT_INDEX], sender_results[TT_FABRIC_WORD_CNT_INDEX + 1]);
        if (sender_bytes != expected_sender_bytes) {
            std::ostringstream message;
            message << "sender " << sender_idx << " reported " << sender_bytes << " bytes, expected "
                    << expected_sender_bytes;
            return make_error(message.str());
        }

        const auto sender_cycles =
            combine_u64(sender_results[TT_FABRIC_CYCLES_INDEX], sender_results[TT_FABRIC_CYCLES_INDEX + 1]);
        if (sender_cycles == 0) {
            std::ostringstream message;
            message << "sender " << sender_idx << " reported zero cycles";
            return make_error(message.str());
        }

        aggregate_bytes += sender_bytes;
        max_sender_cycles = std::max(max_sender_cycles, sender_cycles);
    }

    if (aggregate_bytes != expected_aggregate_bytes) {
        std::ostringstream message;
        message << "aggregate bytes mismatch: got " << aggregate_bytes << ", expected " << expected_aggregate_bytes;
        return make_error(message.str());
    }

    SenderResultSummary result{};
    result.success = true;
    result.aggregate_bytes = aggregate_bytes;
    result.max_sender_cycles = max_sender_cycles;
    return result;
}

}  // namespace tt::tt_fabric::bench
