// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "llrt/rtoptions.hpp"

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

// This test requires simulator environment
TEST_F(QuasarMeshDeviceSingleCardFixture, QuasarComputeKernelMultipleThreads) {
    // Skip if simulator is not available
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    auto mesh_device = devices_[0];

    if (!MetalContext::instance().rtoptions().get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint)) {
        std::cerr << "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to 0,0 to see the output of "
                     "the Compute kernels."
                  << std::endl;
        std::cerr << "WARNING: For example, export TT_METAL_DPRINT_CORES=0,0" << std::endl;
    }

    // We are going to use the first device (0) and the first core (0, 0) on the device.
    const experimental::NodeCoord node{0, 0};
    // Command queue lets us submit work (execute programs and read/write buffers) to the device.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    // Prepare a workload and a device coordinate range that spans the mesh.
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    const uint32_t l1_address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    std::vector<uint32_t> init_values(16, 0);
    tt_metal::detail::WriteToDeviceL1(mesh_device->get_devices()[0], node, l1_address, init_values);

    const experimental::KernelSpecName COMPUTE_KERNEL{"risc_math"};

    experimental::KernelSpec compute_kernel_spec{
        .unique_id = COMPUTE_KERNEL,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/compute/risc_math.cpp",
        .num_threads = 4,
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"l1_address"},
            },
        .hw_config = experimental::ComputeGen2Config{},
    };

    experimental::WorkUnitSpec main_wu{
        .name = "main",
        .kernels = {COMPUTE_KERNEL},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "compute_kernel_multiple_threads",
        .kernels = {compute_kernel_spec},
        .work_units = {main_wu},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = COMPUTE_KERNEL,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(node, {{"l1_address", l1_address}}),
    }};
    experimental::SetProgramRunArgs(program, params);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> actual_values(16, 0);
    tt_metal::detail::ReadFromDeviceL1(
        mesh_device->get_devices()[0], node, l1_address, 16 * sizeof(uint32_t), actual_values);

    const std::vector<uint32_t> expected_values = {4, 6, 5, 9, 8, 10, 9, 13, 12, 14, 13, 17, 16, 18, 17, 21};

    ASSERT_EQ(actual_values, expected_values);
}

// This test requires simulator environment
TEST_F(QuasarMeshDeviceSingleCardFixture, QuasarComputeKernelSingleThread) {
    // Skip if simulator is not available
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    auto mesh_device = devices_[0];

    if (!MetalContext::instance().rtoptions().get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint)) {
        std::cerr << "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to 0,0 to see the output of "
                     "the Compute kernels."
                  << std::endl;
        std::cerr << "WARNING: For example, export TT_METAL_DPRINT_CORES=0,0" << std::endl;
    }

    // We are going to use the first device (0) and the first core (0, 0) on the device.
    const experimental::NodeCoord node{0, 0};
    // Command queue lets us submit work (execute programs and read/write buffers) to the device.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    // Prepare a workload and a device coordinate range that spans the mesh.
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    const uint32_t l1_address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    std::vector<uint32_t> init_values(4, 0);
    tt_metal::detail::WriteToDeviceL1(mesh_device->get_devices()[0], node, l1_address, init_values);

    const experimental::KernelSpecName COMPUTE_KERNEL{"risc_math"};

    experimental::KernelSpec compute_kernel_spec{
        .unique_id = COMPUTE_KERNEL,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/compute/risc_math.cpp",
        .num_threads = 1,
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"l1_address"},
            },
        .hw_config = experimental::ComputeGen2Config{},
    };

    experimental::WorkUnitSpec main_wu{
        .name = "main",
        .kernels = {COMPUTE_KERNEL},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "compute_kernel_single_thread",
        .kernels = {compute_kernel_spec},
        .work_units = {main_wu},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = COMPUTE_KERNEL,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(node, {{"l1_address", l1_address}}),
    }};
    experimental::SetProgramRunArgs(program, params);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> actual_values(4, 0);
    tt_metal::detail::ReadFromDeviceL1(
        mesh_device->get_devices()[0], node, l1_address, 4 * sizeof(uint32_t), actual_values);

    const std::vector<uint32_t> expected_values = {4, 6, 5, 9};

    ASSERT_EQ(actual_values, expected_values);
}

// This test requires simulator environment
TEST_F(QuasarMeshDeviceSingleCardFixture, QuasarCreateMultipleComputeKernelsSingleCluster) {
    // Skip if simulator is not available
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    const experimental::KernelSpecName COMPUTE_KERNEL_1{"risc_math_1"};
    const experimental::KernelSpecName COMPUTE_KERNEL_2{"risc_math_2"};

    experimental::KernelSpec compute_kernel_spec_1{
        .unique_id = COMPUTE_KERNEL_1,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/compute/risc_math.cpp",
        .num_threads = 1,
        .hw_config = experimental::ComputeGen2Config{},
    };

    experimental::KernelSpec compute_kernel_spec_2{
        .unique_id = COMPUTE_KERNEL_2,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/compute/risc_math.cpp",
        .num_threads = 2,
        .hw_config = experimental::ComputeGen2Config{},
    };

    experimental::WorkUnitSpec main_wu{
        .name = "main",
        .kernels = {COMPUTE_KERNEL_1, COMPUTE_KERNEL_2},
        .target_nodes = experimental::NodeCoord{0, 0},
    };

    experimental::ProgramSpec spec{
        .name = "multiple_compute_kernels",
        .kernels = {compute_kernel_spec_1, compute_kernel_spec_2},
        .work_units = {main_wu},
    };

    ASSERT_THROW(experimental::MakeProgramFromSpec(*devices_[0], spec), std::runtime_error);
}
