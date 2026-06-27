#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>

namespace tt::tt_fabric::bench {

using MeshDevicePtr = std::shared_ptr<tt::tt_metal::distributed::MeshDevice>;

struct SenderExecutionContext {
    CoreCoord logical_core;
    uint32_t test_results_address = 0;
};

struct SenderResultSummary {
    bool success = false;
    std::string error_message;
    uint64_t aggregate_bytes = 0;
    uint64_t max_sender_cycles = 0;
};

uint32_t align_up(uint32_t value, uint32_t alignment);
uint32_t to_uint32_checked(size_t value, const char* field_name);
uint64_t combine_u64(uint32_t low, uint32_t high);
uint32_t get_worker_l1_end_address();

std::vector<CoreCoord> enumerate_worker_cores(const MeshDevicePtr& device);

void write_zero_words_to_device(
    tt::tt_metal::IDevice* device, const CoreCoord& logical_core, size_t address, size_t size_bytes);
void write_word_to_device(tt::tt_metal::IDevice* device, const CoreCoord& logical_core, size_t address, uint32_t value);
void initialize_sender_start_barrier_state(
    tt::tt_metal::IDevice* device,
    const CoreCoord& sender_logical_core,
    uint32_t start_signal_address,
    uint32_t ready_count_address,
    uint32_t initial_ready_count);

std::vector<uint32_t> build_sender_noc_xy_encodings(
    const MeshDevicePtr& sender_device, const std::vector<CoreCoord>& sender_logical_cores);
std::vector<uint32_t> build_peer_sender_noc_xy_encodings(
    const std::vector<uint32_t>& sender_noc_xy_encodings, std::size_t sender_idx);

void create_data_movement_kernel(
    tt::tt_metal::Program& program,
    const std::string& kernel_path,
    const CoreCoord& logical_core,
    const std::vector<uint32_t>& compile_args,
    const std::vector<uint32_t>& runtime_args,
    tt::tt_metal::NOC noc = tt::tt_metal::NOC::RISCV_0_default);

void enqueue_single_device_mesh_program(
    const MeshDevicePtr& mesh_device, ChipId physical_device_id, tt::tt_metal::Program&& program);
void enqueue_single_device_mesh_program(const MeshDevicePtr& mesh_device, tt::tt_metal::Program&& program);

SenderResultSummary read_and_validate_sender_results(
    tt::tt_metal::IDevice* device,
    const std::vector<SenderExecutionContext>& sender_execution_contexts,
    uint32_t test_results_size_bytes,
    uint64_t expected_sender_bytes,
    uint64_t expected_aggregate_bytes);

}  // namespace tt::tt_fabric::bench
