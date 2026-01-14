// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "fabric_worker_kernel_helpers.hpp"
#include "fabric_fixture.hpp"
#include "fabric_command_interface.hpp"
#include <stdexcept>
#include <chrono>
#include <llrt/tt_cluster.hpp>

namespace tt::tt_fabric::test_utils {

// Import FabricNodeId into this namespace
using FabricNodeId = tt::tt_fabric::FabricNodeId;

// L1 memory allocation base addresses
// These are used to allocate non-overlapping memory regions for workers
constexpr uint32_t L1_SOURCE_BUFFER_BASE = 0x80000;    // Start at 512KB
constexpr uint32_t L1_BUFFER_SIZE_PER_WORKER = 0x4000; // 16KB per worker
constexpr uint32_t PACKET_PAYLOAD_SIZE_DEFAULT = 256;  // Default packet payload size

WorkerMemoryLayout allocate_worker_memory(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& device) {

    WorkerMemoryLayout layout;

    // Allocate non-overlapping L1 memory regions for this worker
    // Source buffer: where packets are generated
    layout.source_buffer_address = L1_SOURCE_BUFFER_BASE;

    // Teardown signal mailbox: placed after source buffer
    // Allocated as a small region (typically 4 bytes for a uint32_t)
    layout.teardown_signal_address = L1_SOURCE_BUFFER_BASE + 0x1000;

    // Default packet payload size
    layout.packet_payload_size_bytes = PACKET_PAYLOAD_SIZE_DEFAULT;

    return layout;
}

std::shared_ptr<tt_metal::Program> create_traffic_generator_program(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& device,
    const CoreCoord& logical_core,
    const FabricNodeId& dest_fabric_node,
    const WorkerMemoryLayout& mem_layout) {

    auto program = std::make_shared<tt_metal::Program>();

    // Compile-time args must match kernel expectations from CS-003
    // Order: source_buffer_address, packet_payload_size_bytes, target_noc_encoding,
    //        teardown_signal_address, is_2d_fabric
    std::vector<uint32_t> compile_args = {
        mem_layout.source_buffer_address,      // 0: source_buffer_address
        mem_layout.packet_payload_size_bytes,  // 1: packet_payload_size_bytes
        0,                                      // 2: target_noc_encoding (1D fabric = 0)
        mem_layout.teardown_signal_address,    // 3: teardown_signal_address
        0                                       // 4: is_2d_fabric (0 for 1D)
    };

    // Runtime args
    // Order: dest_chip_id, dest_mesh_id, random_seed
    std::vector<uint32_t> runtime_args = {
        dest_fabric_node.mesh_id,   // 0: dest_chip_id (using mesh_id as chip_id)
        0,                           // 1: dest_mesh_id (assuming 0 for default)
        42                           // 2: random_seed
    };

    // Create the traffic generator kernel
    auto kernel_id = tt_metal::CreateKernel(
        *program,
        "tests/tt_metal/tt_fabric/fabric_data_movement/kernels/fabric_traffic_generator.cpp",
        logical_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = compile_args
        });

    // Set runtime args
    tt_metal::SetRuntimeArgs(*program, kernel_id, logical_core, runtime_args);

    return program;
}

void signal_worker_teardown(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& device,
    const CoreCoord& logical_core,
    uint32_t teardown_signal_address) {

    // Write WORKER_TEARDOWN (1) to the teardown signal mailbox in L1
    // This signals the kernel to exit its main loop

    std::vector<uint32_t> data = {WORKER_TEARDOWN};

    // Use Cluster::write_core() to write to L1 on the device
    tt::Cluster::instance().write_core(
        data.data(),
        sizeof(uint32_t),
        logical_core,
        teardown_signal_address,
        device->get_device_id());
}

void wait_for_worker_complete(
    fabric_router_tests::BaseFabricFixture* fixture,
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& device,
    tt_metal::Program& program,
    std::chrono::milliseconds timeout) {

    // Wait for the worker program to complete with timeout
    // This blocks until the program completes or timeout expires

    auto start = std::chrono::steady_clock::now();

    // Call the fixture's wait method to block until program completes
    fixture->WaitForSingleProgramDone(device, program);

    auto elapsed = std::chrono::steady_clock::now() - start;
    if (elapsed > timeout) {
        throw std::runtime_error("Worker kernel did not complete within timeout");
    }
}

} // namespace tt::tt_fabric::test_utils
