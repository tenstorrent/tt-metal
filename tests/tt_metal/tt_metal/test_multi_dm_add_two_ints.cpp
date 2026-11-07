// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt/rtoptions.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

// This test requires simulator environment
TEST_F(QuasarMeshDeviceSingleCardFixture, MultiDmAddTwoInts) {
    // Skip if simulator is not available
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This test can only be run under the simulator or emulator. "
                        "Set TT_METAL_SIMULATOR or TT_METAL_EMULE_MODE=1.";
    }
    auto mesh_device = devices_[0];
    if (mesh_device->compute_with_storage_grid_size().x < 2) {
        GTEST_SKIP() << "This test requires at least 2 worker nodes.";
    }
    if (!MetalContext::instance().rtoptions().get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint)) {
        log_error(
            tt::LogTest,
            "Please set the environment variable TT_METAL_DPRINT_CORES to (0,0),(1,0) to see the output of the Data "
            "Movement kernels.");
        log_error(tt::LogTest, "For example, export TT_METAL_DPRINT_CORES=(0,0),(1,0)");
    }

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    const KernelSpecName KERNEL_0{"kernel_0"};
    const KernelSpecName KERNEL_1{"kernel_1"};
    const KernelSpecName KERNEL_2{"kernel_2"};
    const KernelSpecName KERNEL_3{"kernel_3"};

    const uint32_t result_base = MetalContext::instance().hal().get_dev_addr(
        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);

    auto make_dm_kernel_spec = [](const KernelSpecName& id, uint32_t num_threads, uint32_t l1_addr) {
        return KernelSpec{
            .unique_id = id,
            .source =

                "tests/tt_metal/tt_metal/test_kernels/misc/add_two_ints_2_0.cpp",
            .num_threads = num_threads,
            .compile_time_args = {{"l1_address", l1_addr}},
            .runtime_arg_schema =
                {
                    .runtime_arg_names = {"a", "b"},
                },
            .hw_config = DataMovementHardwareConfig{},
        };
    };

    auto k0 = make_dm_kernel_spec(KERNEL_0, 2, result_base);
    auto k1 = make_dm_kernel_spec(KERNEL_1, 2, result_base + sizeof(int));
    auto k2 = make_dm_kernel_spec(KERNEL_2, 2, result_base + (2 * sizeof(int)));
    auto k3 = make_dm_kernel_spec(KERNEL_3, 2, result_base + (2 * sizeof(int)));

    WorkUnitSpec wu_core0{
        .name = "wu_core0",
        .kernels = {KERNEL_0, KERNEL_1, KERNEL_2},
        .target_nodes = NodeCoord{0, 0},
    };
    WorkUnitSpec wu_core1{
        .name = "wu_core1",
        .kernels = {KERNEL_0, KERNEL_1, KERNEL_3},
        .target_nodes = NodeCoord{1, 0},
    };

    ProgramSpec spec{
        .name = "multi_dm_add_two_ints",
        .kernels = {k0, k1, k2, k3},
        .work_units = {wu_core0, wu_core1},
    };
    Program program = tt_metal::experimental::MakeProgramFromSpec(*mesh_device, spec);

    ProgramRunArgs params;
    params.kernel_run_args = {
        ProgramRunArgs::KernelRunArgs{
            .kernel = KERNEL_0,
            .runtime_arg_values =
                {{"a", {{NodeCoord{0, 0}, 1}, {NodeCoord{1, 0}, 1}}},
                 {"b", {{NodeCoord{0, 0}, 2}, {NodeCoord{1, 0}, 2}}}}},
        ProgramRunArgs::KernelRunArgs{
            .kernel = KERNEL_1,
            .runtime_arg_values =
                {{"a", {{NodeCoord{0, 0}, 3}, {NodeCoord{1, 0}, 3}}},
                 {"b", {{NodeCoord{0, 0}, 4}, {NodeCoord{1, 0}, 4}}}}},
        ProgramRunArgs::KernelRunArgs{
            .kernel = KERNEL_2,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(NodeCoord{0, 0}, {{"a", 5}, {"b", 6}})},
        ProgramRunArgs::KernelRunArgs{
            .kernel = KERNEL_3,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(NodeCoord{1, 0}, {{"a", 7}, {"b", 8}})},
    };
    tt_metal::experimental::SetProgramRunArgs(program, params);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> result_core_0(3, 0);
    slow_dispatch::ReadFromL1(this->device(), CoreCoord(0, 0), result_base, sizeof(uint32_t) * 3, result_core_0);

    std::vector<uint32_t> result_core_1(3, 0);
    slow_dispatch::ReadFromL1(this->device(), CoreCoord(1, 0), result_base, sizeof(uint32_t) * 3, result_core_1);

    ASSERT_EQ(result_core_0, (std::vector<uint32_t>{3, 7, 11}));
    ASSERT_EQ(result_core_1, (std::vector<uint32_t>{3, 7, 15}));
}
