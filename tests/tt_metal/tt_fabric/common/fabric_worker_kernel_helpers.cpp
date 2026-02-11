// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "fabric_worker_kernel_helpers.hpp"
#include "fabric_fixture.hpp"
#include "fabric_command_interface.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tt_align.hpp>
#include <stdexcept>
#include <chrono>
#include <llrt/tt_cluster.hpp>
#include "tests/tt_metal/tt_fabric/common/fabric_worker_kernel_helpers.hpp"

namespace tt::tt_fabric::test_utils {

// Import FabricNodeId into this namespace
using FabricNodeId = tt::tt_fabric::FabricNodeId;

// Default packet payload size for traffic generation
constexpr uint32_t PACKET_PAYLOAD_SIZE_DEFAULT = 256;

// Memory region sizes
constexpr uint32_t SOURCE_BUFFER_SIZE = 0x1000;  // 4KB for source buffer

WorkerMemoryLayout allocate_worker_memory() {

    const auto& hal = tt::tt_metal::MetalContext::instance().hal();

    // Get the unreserved L1 base address for Tensix cores from HAL
    uint32_t l1_unreserved_base = hal.get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::DEFAULT_UNRESERVED);

    // Get L1 alignment requirement
    size_t l1_alignment = hal.get_alignment(tt::tt_metal::HalMemType::L1);

    // Calculate addresses before struct initialization to avoid using uninitialized values
    uint32_t source_buffer_addr = tt::align(l1_unreserved_base, l1_alignment);
    uint32_t teardown_signal_addr = tt::align(source_buffer_addr + SOURCE_BUFFER_SIZE, l1_alignment);

    WorkerMemoryLayout layout = {
        .source_buffer_address = source_buffer_addr,
        .teardown_signal_address = teardown_signal_addr,
        .packet_payload_size_bytes = PACKET_PAYLOAD_SIZE_DEFAULT};

    return layout;
}

std::shared_ptr<tt_metal::Program> create_traffic_generator_program(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& device,
    const CoreCoord& logical_core,
    const FabricNodeId& dest_fabric_node,
    const WorkerMemoryLayout& mem_layout) {

    auto program = std::make_shared<tt_metal::Program>();

    // Get source fabric node ID from the mesh device (use first device at coord 0,0)
    FabricNodeId src_fabric_node(MeshId{0}, 0);

    // Target core on remote chip for traffic destination
    CoreCoord remote_logical_core(0, 0);

    // Get remote buffer address from HAL (use unreserved L1 space on remote core)
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    uint32_t remote_buffer_addr = hal.get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::TENSIX, tt::tt_metal::HalL1MemAddrType::DEFAULT_UNRESERVED);

    // Get the physical NOC coords for the remote core
    auto physical_core = device->worker_core_from_logical_core(remote_logical_core);
    uint32_t virt_noc_x = physical_core.x;
    uint32_t virt_noc_y = physical_core.y;

    // Compile-time args for the kernel
    std::vector<uint32_t> compile_args = {
        mem_layout.source_buffer_address,      // 0: source_buffer_address
        mem_layout.packet_payload_size_bytes,  // 1: packet_payload_size_bytes
        mem_layout.teardown_signal_address,    // 2: teardown_signal_address
        virt_noc_x,                             // 3: virt_noc_x
        virt_noc_y,                             // 4: virt_noc_y
        remote_buffer_addr,                     // 5: remote_buffer_addr
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

    // Runtime args: start with basic args, then append fabric connection
    std::vector<uint32_t> runtime_args = {
        dest_fabric_node.chip_id,        // 0: dest_chip_id
        dest_fabric_node.mesh_id.get(),  // 1: dest_mesh_id
    };

    // Get a valid link index for the connection
    auto link_indices = tt::tt_fabric::get_forwarding_link_indices(src_fabric_node, dest_fabric_node);
    TT_FATAL(!link_indices.empty(), "No forwarding link indices found for source {} to destination {}", src_fabric_node, dest_fabric_node);

    // Append fabric connection runtime args
    tt::tt_fabric::append_fabric_connection_rt_args(
        src_fabric_node,
        dest_fabric_node,
        0,
        *program,
        logical_core,
        runtime_args,
        CoreType::WORKER);

    // Set runtime args
    tt_metal::SetRuntimeArgs(*program, kernel_id, logical_core, runtime_args);

    return program;
}

void signal_worker_teardown(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& mesh_device,
    const CoreCoord& logical_core,
    uint32_t teardown_signal_address) {

    // Write WORKER_TEARDOWN (1) to the teardown signal mailbox in L1
    // This signals the kernel to exit its main loop

    std::vector<uint32_t> data = {WORKER_TEARDOWN};

    // Get the actual device where the kernel is running (first device in mesh)
    auto* device = mesh_device->get_devices()[0];

    auto virtual_core = mesh_device->virtual_core_from_logical_core(logical_core, CoreType::WORKER);

    // Use Cluster::write_core() to write to L1 on the device
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        data.data(),
        sizeof(uint32_t),
        tt_cxy_pair(device->id(), virtual_core),
        teardown_signal_address);
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
