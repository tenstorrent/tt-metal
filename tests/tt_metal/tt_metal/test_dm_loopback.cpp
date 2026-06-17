// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

// This test requires simulator environment
TEST_F(QuasarMeshDeviceSingleCardFixture, DmLoopback) {
    // Skip if simulator is not available
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

    IDevice* dev = devices_[0]->get_devices()[0];
    auto mesh_device = devices_[0];
    const experimental::NodeCoord node{0, 0};

    // These addresses have been randomly chosen
    uint32_t l1_address = 1000 * 1024;
    uint32_t dram_address = 30000 * 1024;
    std::vector<uint32_t> value = {0x12345678};

    tt_metal::detail::WriteToDeviceDRAMChannel(dev, 0, dram_address, value);
    MetalContext::instance().get_cluster().dram_barrier(dev->id());

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    // Metal 2.0 reserves DM0/DM1; max 6 user DM threads per node.
    // Reduced from 4+4=8 to 3+3=6 loopback stages to fit within the limit.
    constexpr uint32_t num_loopback_stages = 3;

    const experimental::KernelSpecName DRAM_TO_L1_0{"dram_to_l1_0"};
    const experimental::KernelSpecName DRAM_TO_L1_1{"dram_to_l1_1"};
    const experimental::KernelSpecName DRAM_TO_L1_2{"dram_to_l1_2"};
    const experimental::KernelSpecName L1_TO_DRAM_0{"l1_to_dram_0"};
    const experimental::KernelSpecName L1_TO_DRAM_1{"l1_to_dram_1"};
    const experimental::KernelSpecName L1_TO_DRAM_2{"l1_to_dram_2"};

    auto make_dram_to_l1_spec = [](const experimental::KernelSpecName& id) {
        return experimental::KernelSpec{
            .unique_id = id,
            .source =

                OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_to_l1.cpp",
            .num_threads = 1,
            .semaphore_bindings =
                {{.semaphore_spec_name = experimental::SemaphoreSpecName{"sem"}, .accessor_name = "sem"}},
            .runtime_arg_schema =
                {
                    .runtime_arg_names = {"dram_addr", "l1_addr", "dram_buffer_size", "dram_bank_id", "signal_value"},
                },
            .hw_config =
                experimental::DataMovementHardwareConfig{
                    .gen2_config = experimental::DataMovementHardwareConfig::Gen2Config{}},
        };
    };

    auto make_l1_to_dram_spec = [](const experimental::KernelSpecName& id) {
        return experimental::KernelSpec{
            .unique_id = id,
            .source =

                OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/l1_to_dram.cpp",
            .num_threads = 1,
            .semaphore_bindings =
                {{.semaphore_spec_name = experimental::SemaphoreSpecName{"sem"}, .accessor_name = "sem"}},
            .runtime_arg_schema =
                {
                    .runtime_arg_names = {"dram_addr", "l1_addr", "dram_buffer_size", "dram_bank_id", "signal_value"},
                },
            .hw_config =
                experimental::DataMovementHardwareConfig{
                    .gen2_config = experimental::DataMovementHardwareConfig::Gen2Config{}},
        };
    };

    experimental::SemaphoreSpec sem{
        .unique_id = experimental::SemaphoreSpecName{"sem"},
        .target_nodes = node,
    };

    experimental::WorkUnitSpec main_wu{
        .name = "main",
        .kernels = {DRAM_TO_L1_0, DRAM_TO_L1_1, DRAM_TO_L1_2, L1_TO_DRAM_0, L1_TO_DRAM_1, L1_TO_DRAM_2},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "dm_loopback",
        .kernels =
            {make_dram_to_l1_spec(DRAM_TO_L1_0),
             make_dram_to_l1_spec(DRAM_TO_L1_1),
             make_dram_to_l1_spec(DRAM_TO_L1_2),
             make_l1_to_dram_spec(L1_TO_DRAM_0),
             make_l1_to_dram_spec(L1_TO_DRAM_1),
             make_l1_to_dram_spec(L1_TO_DRAM_2)},
        .semaphores = {sem},
        .work_units = {main_wu},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    const experimental::KernelSpecName dram_to_l1_names[] = {DRAM_TO_L1_0, DRAM_TO_L1_1, DRAM_TO_L1_2};
    const experimental::KernelSpecName l1_to_dram_names[] = {L1_TO_DRAM_0, L1_TO_DRAM_1, L1_TO_DRAM_2};

    experimental::ProgramRunArgs params;
    uint32_t signal_value = 0;
    for (uint32_t i = 0; i < num_loopback_stages; i++) {
        params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = dram_to_l1_names[i],
            .runtime_arg_values = {
                {node,
                 {{"dram_addr", dram_address},
                  {"l1_addr", l1_address},
                  {"dram_buffer_size", 4u},
                  {"dram_bank_id", 0u},
                  {"signal_value", signal_value}}}}});
        dram_address += 1024;
        signal_value++;

        params.kernel_run_args.push_back(experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = l1_to_dram_names[i],
            .runtime_arg_values = {
                {node,
                 {{"dram_addr", dram_address},
                  {"l1_addr", l1_address},
                  {"dram_buffer_size", 4u},
                  {"dram_bank_id", 0u},
                  {"signal_value", signal_value}}}}});
        l1_address += sizeof(uint32_t);
        signal_value++;
    }
    experimental::SetProgramRunArgs(program, params);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> outputs{0};
    tt_metal::detail::ReadFromDeviceDRAMChannel(dev, 0, dram_address, sizeof(uint32_t), outputs);

    ASSERT_EQ(outputs[0], value[0]) << "Got the value " << std::hex << outputs[0] << " instead of " << value[0];
}
