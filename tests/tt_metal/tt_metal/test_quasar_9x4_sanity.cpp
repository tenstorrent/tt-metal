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

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

TEST_F(QuasarMeshDeviceSingleCardFixture, Quasar9x4SanityAllCoresAck) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }

    IDevice* dev = devices_[0]->get_devices()[0];
    auto mesh_device = devices_[0];
    auto grid = dev->compute_with_storage_grid_size();
    log_info(LogTest, "Compute grid: {}x{}", grid.x, grid.y);

    constexpr uint32_t SEM_L1_ADDR = 100 * 1024;
    const uint32_t total_cores = grid.x * grid.y;
    const uint32_t expected_acks = total_cores - 1;

    const experimental::metal2_host_api::NodeCoord leader{0, 0};
    std::vector<uint32_t> zero{0};
    tt_metal::detail::WriteToDeviceL1(dev, leader, SEM_L1_ADDR, zero);
    MetalContext::instance().get_cluster().l1_barrier(dev->id());

    const CoreCoord leader_phys = mesh_device->worker_core_from_logical_core(leader);

    constexpr const char* ACK_KERNEL = "ack_kernel";
    experimental::metal2_host_api::KernelSpec ack_kernel_spec{
        .unique_id = ACK_KERNEL,
        .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/quasar_9x4_ack.cpp",
        .num_threads = 1,
        .runtime_arguments_schema =
            {
                .named_runtime_args =
                    {"is_leader", "my_x", "my_y", "leader_noc_x", "leader_noc_y", "expected_acks", "sem_addr"},
            },
        .config_spec =
            experimental::metal2_host_api::DataMovementConfiguration{
                .gen2_data_movement_config =
                    experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
    };

    experimental::metal2_host_api::NodeRange all_cores{
        experimental::metal2_host_api::NodeCoord{0, 0},
        experimental::metal2_host_api::NodeCoord{grid.x - 1, grid.y - 1}};

    experimental::metal2_host_api::WorkUnitSpec wu{
        .unique_id = "all_cores",
        .kernels = {ACK_KERNEL},
        .target_nodes = all_cores,
    };

    experimental::metal2_host_api::ProgramSpec spec{
        .program_id = "quasar_9x4_sanity",
        .kernels = {ack_kernel_spec},
        .work_units = {wu},
    };
    Program program = experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, spec);

    experimental::metal2_host_api::ProgramRunParams params;
    experimental::metal2_host_api::ProgramRunParams::KernelRunParams krp;
    krp.kernel_spec_name = ACK_KERNEL;
    for (uint32_t y = 0; y < grid.y; ++y) {
        for (uint32_t x = 0; x < grid.x; ++x) {
            uint32_t is_leader = (x == 0 && y == 0) ? 1u : 0u;
            krp.named_runtime_args.push_back({
                .node = experimental::metal2_host_api::NodeCoord{x, y},
                .args =
                    {
                        {"is_leader", is_leader},
                        {"my_x", x},
                        {"my_y", y},
                        {"leader_noc_x", static_cast<uint32_t>(leader_phys.x)},
                        {"leader_noc_y", static_cast<uint32_t>(leader_phys.y)},
                        {"expected_acks", expected_acks},
                        {"sem_addr", SEM_L1_ADDR},
                    },
            });
        }
    }
    params.kernel_run_params.push_back(std::move(krp));
    experimental::metal2_host_api::SetProgramRunParameters(program, params);

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> sem_value(1, 0);
    tt_metal::detail::ReadFromDeviceL1(dev, leader, SEM_L1_ADDR, sizeof(uint32_t), sem_value);
    log_info(LogTest, "Leader semaphore final value: {} (expected {})", sem_value[0], expected_acks);
    ASSERT_EQ(sem_value[0], expected_acks);
}
