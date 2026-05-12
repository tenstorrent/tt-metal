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
#include "hw/inc/internal/tt-2xx/quasar/dev_mem_map.h"
#include "llrt/rtoptions.hpp"

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
    if (!MetalContext::instance().rtoptions().get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint)) {
        log_error(
            tt::LogTest,
            "Please set the environment variable TT_METAL_DPRINT_CORES to (0,0),(1,0) to see the output of the Data "
            "Movement kernels.");
        log_error(tt::LogTest, "For example, export TT_METAL_DPRINT_CORES=(0,0),(1,0)");
    }

    IDevice* dev = devices_[0]->get_devices()[0];
    auto mesh_device = devices_[0];

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    constexpr const char* KERNEL_0 = "kernel_0";
    constexpr const char* KERNEL_1 = "kernel_1";
    constexpr const char* KERNEL_2 = "kernel_2";
    constexpr const char* KERNEL_3 = "kernel_3";

    auto make_dm_kernel_spec = [](const char* id, uint8_t num_threads, uint32_t l1_addr) {
        return experimental::metal2_host_api::KernelSpec{
            .unique_id = id,
            .source =
                experimental::metal2_host_api::KernelSpec::SourceFilePath{
                    "tests/tt_metal/tt_metal/test_kernels/misc/add_two_ints.cpp"},
            .num_threads = num_threads,
            .compile_time_arg_bindings = {{"l1_address", l1_addr}},
            .runtime_arguments_schema =
                {
                    .named_runtime_args = {"a", "b"},
                },
            .config_spec =
                experimental::metal2_host_api::DataMovementConfiguration{
                    .gen2_data_movement_config =
                        experimental::metal2_host_api::DataMovementConfiguration::Gen2DataMovementConfig{}},
        };
    };

    auto k0 = make_dm_kernel_spec(KERNEL_0, 2, MEM_L1_UNCACHED_BASE);
    auto k1 = make_dm_kernel_spec(KERNEL_1, 2, MEM_L1_UNCACHED_BASE + sizeof(int));
    auto k2 = make_dm_kernel_spec(KERNEL_2, 2, MEM_L1_UNCACHED_BASE + (2 * sizeof(int)));
    auto k3 = make_dm_kernel_spec(KERNEL_3, 2, MEM_L1_UNCACHED_BASE + (2 * sizeof(int)));

    experimental::metal2_host_api::WorkUnitSpec wu_core0{
        .unique_id = "wu_core0",
        .kernels = {KERNEL_0, KERNEL_1, KERNEL_2},
        .target_nodes = experimental::metal2_host_api::NodeCoord{0, 0},
    };
    experimental::metal2_host_api::WorkUnitSpec wu_core1{
        .unique_id = "wu_core1",
        .kernels = {KERNEL_0, KERNEL_1, KERNEL_3},
        .target_nodes = experimental::metal2_host_api::NodeCoord{1, 0},
    };

    experimental::metal2_host_api::ProgramSpec spec{
        .program_id = "multi_dm_add_two_ints",
        .kernels = {k0, k1, k2, k3},
        .work_units = {wu_core0, wu_core1},
    };
    Program program = experimental::metal2_host_api::MakeProgramFromSpec(*mesh_device, spec);

    experimental::metal2_host_api::ProgramRunParams params;
    params.kernel_run_params = {
        {.kernel_spec_name = KERNEL_0,
         .named_runtime_args =
             {{.node = experimental::metal2_host_api::NodeCoord{0, 0}, .args = {{"a", 1}, {"b", 2}}},
              {.node = experimental::metal2_host_api::NodeCoord{1, 0}, .args = {{"a", 1}, {"b", 2}}}}},
        {.kernel_spec_name = KERNEL_1,
         .named_runtime_args =
             {{.node = experimental::metal2_host_api::NodeCoord{0, 0}, .args = {{"a", 3}, {"b", 4}}},
              {.node = experimental::metal2_host_api::NodeCoord{1, 0}, .args = {{"a", 3}, {"b", 4}}}}},
        {.kernel_spec_name = KERNEL_2,
         .named_runtime_args =
             {{.node = experimental::metal2_host_api::NodeCoord{0, 0}, .args = {{"a", 5}, {"b", 6}}}}},
        {.kernel_spec_name = KERNEL_3,
         .named_runtime_args =
             {{.node = experimental::metal2_host_api::NodeCoord{1, 0}, .args = {{"a", 7}, {"b", 8}}}}},
    };
    experimental::metal2_host_api::SetProgramRunParameters(program, params);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> result_core_0(3, 0);
    tt_metal::detail::ReadFromDeviceL1(dev, CoreCoord(0, 0), 0, sizeof(uint32_t) * 3, result_core_0);

    std::vector<uint32_t> result_core_1(3, 0);
    tt_metal::detail::ReadFromDeviceL1(dev, CoreCoord(1, 0), 0, sizeof(uint32_t) * 3, result_core_1);

    ASSERT_EQ(result_core_0, (std::vector<uint32_t>{3, 7, 11}));
    ASSERT_EQ(result_core_1, (std::vector<uint32_t>{3, 7, 15}));
}
