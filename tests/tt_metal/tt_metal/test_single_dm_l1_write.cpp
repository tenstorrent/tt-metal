// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

// This test requires simulator environment
TEST_F(MeshDeviceSingleCardFixture, SingleDmL1Write) {
    // Skip if simulator is not available
    char* env_var = std::getenv("TT_METAL_SIMULATOR");
    if (env_var == nullptr) {
        GTEST_SKIP() << "This test can only be run using a simulator. Set TT_METAL_SIMULATOR environment variable.";
    }

    IDevice* dev = devices_[0]->get_devices()[0];
    auto mesh_device = devices_[0];

    const uint32_t address = 100 * 1024;
    const uint32_t value = 0x12345678;
    const std::unordered_map<std::string, uint32_t> named_compile_time_args = {
        {"buffer_size", 1024},
        {"", 3},
        {"!@#$%^&*()", 12},
        {"very_long_parameter_name_that_someone_could_potentially_use_to_try_to_break_the_kernel", 456}};
    std::vector<uint32_t> outputs(1);
    outputs[0] = 0;
    env_var = std::getenv("TT_METAL_DPRINT_CORES");
    if (env_var == nullptr) {
        std::cerr << "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to 0,0 to see the output of "
                     "the Data Movement kernels."
                  << std::endl;
        std::cerr << "WARNING: For example, export TT_METAL_DPRINT_CORES=0,0" << std::endl;
    }

    // We are going to use the first device (0) and the first core (0, 0) on the device.
    constexpr CoreCoord core = {0, 0};
    tt_metal::detail::WriteToDeviceL1(dev, core, address, outputs);
    // Command queue lets us submit work (execute programs and read/write buffers) to the device.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    // Prepare a workload and a device coordinate range that spans the mesh.
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    // Configure and create Data Movement kernel
    // Quasar currently supports only one Data Movement core.
    KernelHandle data_movement_kernel_0 = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/simple_l1_write.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .named_compile_args = named_compile_time_args});

    // Set Runtime Arguments for the Data Movement Kernel (memory address to write to)
    SetRuntimeArgs(program, data_movement_kernel_0, core, {address});
    SetCommonRuntimeArgs(program, data_movement_kernel_0, {value});
    std::cout << "Hello, Core {0, 0} on Device 0, Please start execution. I will standby for your communication."
              << std::endl;

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);
    tt_metal::detail::ReadFromDeviceL1(dev, core, address, 4, outputs);

    ASSERT_EQ(outputs[0], value) << "Got the value " << std::hex << outputs[0] << " instead of " << value;
}
