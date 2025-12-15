// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main() {
    using namespace tt;
    using namespace tt::tt_metal;
    std::cout << "Test started" << std::endl;

    char* env_var = std::getenv("TT_METAL_SIMULATOR");
    if (env_var == nullptr) {
        std::cerr
            << "ERROR: This test can only be run using a simulator. Please set Environment Variable TT_METAL_SIMULATOR"
            << std::endl;
        std::cerr << "ERROR: with a valid simulator path" << std::endl;
        return 1;
    }
    env_var = std::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (env_var == nullptr) {
        std::cerr << "ERROR: This test can only be run in slow dispatch mode. Please set Environment Variable "
                     "TT_METAL_SLOW_DISPATCH_MODE"
                  << std::endl;
        std::cerr << "ERROR: using export TT_METAL_SLOW_DISPATCH_MODE=1" << std::endl;
        return 1;
    }

    constexpr uint32_t l1_read_write_address = 100 * 1024;
    constexpr uint32_t dram_read_address = 30000 * 1024;
    constexpr uint32_t dram_write_address = 40000 * 1024;
    std::vector<uint32_t> value = {0x12345678};

    // Initialize mesh device (1x1), command queue, workload, device range, and program.
    // We are going to use the first device (0) and the first core (0, 0) on the device.
    constexpr CoreCoord core = {0, 0};
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    tt_metal::detail::WriteToDeviceDRAMChannel(mesh_device->get_devices()[0], 0, dram_read_address, value);
    std::cout << "WriteToDeviceDRAMChannel passed" << std::endl;
    MetalContext::instance().get_cluster().dram_barrier(mesh_device->get_devices()[0]->id());
    std::cout << "DRAM barrier passed" << std::endl;

    // Command queue lets us submit work (execute programs and read/write buffers) to the device.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    // Prepare a workload and a device coordinate range that spans the mesh.
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    // Configure and create Data Movement kernels
    KernelHandle data_movement_kernel_0 = experimental::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_to_l1.cpp",
        core,
        experimental::QuasarDataMovementConfig{.processors = {DataMovementProcessor::RISCV_0}});

    KernelHandle data_movement_kernel_1 = experimental::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/l1_to_dram.cpp",
        core,
        experimental::QuasarDataMovementConfig{.processors = {DataMovementProcessor::RISCV_1}});

    const uint32_t sem_id = CreateSemaphore(program, core, 0);

    SetRuntimeArgs(program, data_movement_kernel_0, core, {dram_write_address, l1_read_write_address, 4, 0, sem_id});
    SetRuntimeArgs(program, data_movement_kernel_1, core, {dram_read_address, l1_read_write_address, 4, 0, sem_id});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);
    std::cout << "EnqueueMeshWorkload passed" << std::endl;
    std::vector<uint32_t> outputs{0};
    tt_metal::detail::ReadFromDeviceDRAMChannel(mesh_device->get_devices()[0], 0, dram_write_address, 4, outputs);
    std::cout << "ReadFromDeviceDRAMChannel passed" << std::endl;
    mesh_device->close();

    if (outputs[0] == value[0]) {
        std::cout << "Test passed!" << std::endl;
        return 0;
    } else {
        std::cout << "Test failed! Got the value " << std::hex << outputs[0] << " instead of " << value[0] << std::endl;
        return 1;
    }
}