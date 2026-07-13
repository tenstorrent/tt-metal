// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"
#include "context/metal_context.hpp"

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

// This test requires simulator environment
TEST_F(QuasarMeshDeviceSingleCardFixture, SingleDmL1Write) {
    // Skip if simulator is not available
    char* env_var = std::getenv("TT_METAL_SIMULATOR");
    if (env_var == nullptr) {
        GTEST_SKIP() << "This test can only be run using a simulator. Set TT_METAL_SIMULATOR environment variable.";
    }

    IDevice* dev = devices_[0]->get_devices()[0];
    auto mesh_device = devices_[0];

    const uint32_t address = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    const uint32_t value = 0x12345678;
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
    const experimental::NodeCoord node{0, 0};
    tt_metal::detail::WriteToDeviceL1(dev, node, address, outputs);
    // Command queue lets us submit work (execute programs and read/write buffers) to the device.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    // Prepare a workload and a device coordinate range that spans the mesh.
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    const experimental::KernelSpecName DM_KERNEL{"dm_kernel"};

    experimental::KernelSpec dm_kernel_spec{
        .unique_id = DM_KERNEL,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/simple_l1_write.cpp",
        .num_threads = 2,
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"address"},
                .common_runtime_arg_names = {"value"},
            },
        .hw_config = experimental::DataMovementGen2Config{},
    };

    experimental::WorkUnitSpec main_wu{
        .name = "main",
        .kernels = {DM_KERNEL},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "single_dm_l1_write",
        .kernels = {dm_kernel_spec},
        .work_units = {main_wu},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = DM_KERNEL,
        .runtime_arg_values = {{node, {{"address", address}}}},
        .common_runtime_arg_values = {{"value", value}},
    }};
    experimental::SetProgramRunArgs(program, params);
    std::cout << "Hello, Core {0, 0} on Device 0, Please start execution. I will standby for your communication."
              << std::endl;

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);
    tt_metal::detail::ReadFromDeviceL1(dev, node, address, 4, outputs);

    ASSERT_EQ(outputs[0], value) << "Got the value " << std::hex << outputs[0] << " instead of " << value;
}
