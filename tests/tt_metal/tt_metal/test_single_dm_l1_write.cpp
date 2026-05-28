// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

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

    const uint32_t address = 100 * 1024;
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
    const experimental::metal2_host_api::NodeCoord node{0, 0};
    tt_metal::detail::WriteToDeviceL1(dev, node, address, outputs);
    // Command queue lets us submit work (execute programs and read/write buffers) to the device.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    // Prepare a workload and a device coordinate range that spans the mesh.
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    constexpr const char* DM_KERNEL = "dm_kernel";

    experimental::metal2_host_api::KernelSpec dm_kernel_spec{
        .unique_id = DM_KERNEL,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/simple_l1_write.cpp",
        .num_threads = 2,
        .runtime_arguments_schema =
            {
                .named_runtime_args = {"address"},
                .named_common_runtime_args = {"value"},
            },
        .config_spec =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };

    experimental::metal2_host_api::WorkUnitSpec main_wu{
        .unique_id = "main",
        .kernels = {DM_KERNEL},
        .target_nodes = node,
    };

    experimental::metal2_host_api::ProgramSpec spec{
        .program_id = "single_dm_l1_write",
        .kernels = {dm_kernel_spec},
        .work_units = {main_wu},
    };
    Program program = experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, spec);

    experimental::metal2_host_api::ProgramRunParams params;
    params.kernel_run_params = {{
        .kernel_spec_name = DM_KERNEL,
        .named_runtime_args = {{.node = node, .args = {{"address", address}}}},
        .named_common_runtime_args = {{"value", value}},
    }};
    experimental::metal2_host_api::SetProgramRunParameters(program, params);
    std::cout << "Hello, Core {0, 0} on Device 0, Please start execution. I will standby for your communication."
              << std::endl;

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);
    tt_metal::detail::ReadFromDeviceL1(dev, node, address, 4, outputs);

    ASSERT_EQ(outputs[0], value) << "Got the value " << std::hex << outputs[0] << " instead of " << value;
}
