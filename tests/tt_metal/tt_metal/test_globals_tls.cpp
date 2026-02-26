// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

// This test requires simulator environment
TEST_F(MeshDeviceSingleCardFixture, GlobalsAndTLS) {
//    IDevice* dev = devices_[0]->get_devices()[0];
    tt::tt_metal::MetalContext::instance().rtoptions().set_force_jit_compile(true);
    auto mesh_device = devices_[0];

    const uint32_t signal_address = 100 * 1024;
    const uint32_t dram_address = 30000 * 1024;

    std::vector<uint32_t> outputs(1);
    outputs[0] = 0;
    char* env_var = std::getenv("TT_METAL_SIMULATOR");
    if (env_var == nullptr) {
        GTEST_SKIP() << "This test can only be run using a simulator. Set TT_METAL_SIMULATOR environment variable.";
    }
    env_var = std::getenv("TT_METAL_DPRINT_CORES");
    if (env_var == nullptr) {
        std::cerr << "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to 0,0 to see the output of "
                     "the Data Movement kernels."
                  << std::endl;
        std::cerr << "WARNING: For example, export TT_METAL_DPRINT_CORES=0,0" << std::endl;
    }

    // We are going to use the first device (0) and the first core (0, 0) on the device.
    constexpr CoreCoord core = {0, 0};
    // Command queue lets us submit work (execute programs and read/write buffers) to the device.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    // Prepare a workload and a device coordinate range that spans the mesh.
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    // Configure and create Data Movement kernel
    KernelHandle data_movement_kernel_0 = experimental::quasar::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/simple_tls_check_1.cpp",
        core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_processors_per_cluster = 4});

    KernelHandle data_movement_kernel_1 = experimental::quasar::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/simple_tls_check_2.cpp",
        core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_processors_per_cluster = 3});
            
    KernelHandle data_movement_kernel_2 = experimental::quasar::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/simple_tls_check_3.cpp",
        core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_processors_per_cluster = 1});

    // Set Runtime Arguments for the Data Movement Kernel (memory address to write to)
    SetRuntimeArgs(program, data_movement_kernel_0, core, {signal_address, dram_address});
    SetRuntimeArgs(program, data_movement_kernel_1, core, {signal_address, dram_address});
    SetRuntimeArgs(program, data_movement_kernel_2, core, {signal_address, dram_address});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);
//    tt_metal::detail::ReadFromDeviceL1(dev, core, address, 4, outputs);
}