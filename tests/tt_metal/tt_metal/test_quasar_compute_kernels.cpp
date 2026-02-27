// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "llrt/rtoptions.hpp"

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

// This test requires simulator environment
TEST_F(MeshDeviceSingleCardFixture, QuasarComputeKernelMultipleThreads) {
    // Skip if simulator is not available
    if (!MetalContext::instance().rtoptions().get_simulator_enabled()) {
        GTEST_SKIP() << "This test can only be run using a simulator. Set TT_METAL_SIMULATOR environment variable.";
    }

    auto mesh_device = devices_[0];

    if (!MetalContext::instance().rtoptions().get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint)) {
        std::cerr << "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to 0,0 to see the output of "
                     "the Compute kernels."
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

    constexpr uint32_t l1_address = 1000 * 1024;
    std::vector<uint32_t> init_values(16, 0);
    tt_metal::detail::WriteToDeviceL1(mesh_device->get_devices()[0], core, l1_address, init_values);

    const KernelHandle risc_math_kernel = experimental::quasar::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/compute/risc_math.cpp",
        core,
        experimental::quasar::QuasarComputeConfig{.num_threads_per_cluster = 4});
    SetRuntimeArgs(program, risc_math_kernel, core, {l1_address});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> actual_values(16, 0);
    tt_metal::detail::ReadFromDeviceL1(
        mesh_device->get_devices()[0], core, l1_address, 16 * sizeof(uint32_t), actual_values);

    const std::vector<uint32_t> expected_values = {4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15, 16, 18, 17, 19};

    ASSERT_EQ(actual_values, expected_values);
}

// This test requires simulator environment
TEST_F(MeshDeviceSingleCardFixture, QuasarComputeKernelSingleThread) {
    // Skip if simulator is not available
    if (!MetalContext::instance().rtoptions().get_simulator_enabled()) {
        GTEST_SKIP() << "This test can only be run using a simulator. Set TT_METAL_SIMULATOR environment variable.";
    }

    auto mesh_device = devices_[0];

    if (!MetalContext::instance().rtoptions().get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint)) {
        std::cerr << "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to 0,0 to see the output of "
                     "the Compute kernels."
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

    constexpr uint32_t l1_address = 1000 * 1024;
    std::vector<uint32_t> init_values(4, 0);
    tt_metal::detail::WriteToDeviceL1(mesh_device->get_devices()[0], core, l1_address, init_values);

    const KernelHandle risc_math_kernel = experimental::quasar::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/compute/risc_math.cpp",
        core,
        experimental::quasar::QuasarComputeConfig{.num_threads_per_cluster = 1});
    SetRuntimeArgs(program, risc_math_kernel, core, {l1_address});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> actual_values(4, 0);
    tt_metal::detail::ReadFromDeviceL1(
        mesh_device->get_devices()[0], core, l1_address, 4 * sizeof(uint32_t), actual_values);

    const std::vector<uint32_t> expected_values = {4, 6, 5, 7};

    ASSERT_EQ(actual_values, expected_values);
}

// This test requires simulator environment
TEST_F(MeshDeviceSingleCardFixture, QuasarCreateMultipleComputeKernelsSingleCluster) {
    // Skip if simulator is not available
    if (!MetalContext::instance().rtoptions().get_simulator_enabled()) {
        GTEST_SKIP() << "This test can only be run using a simulator. Set TT_METAL_SIMULATOR environment variable.";
    }

    constexpr CoreCoord core = {0, 0};
    Program program = CreateProgram();

    experimental::quasar::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/compute/risc_math.cpp",
        core,
        experimental::quasar::QuasarComputeConfig{.num_threads_per_cluster = 1});

    ASSERT_THROW(
        experimental::quasar::CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/compute/risc_math.cpp",
            core,
            experimental::quasar::QuasarComputeConfig{.num_threads_per_cluster = 2}),
        std::runtime_error);
}
