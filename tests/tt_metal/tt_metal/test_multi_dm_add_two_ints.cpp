// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "hw/inc/internal/tt-2xx/quasar/dev_mem_map.h"

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

// This test requires simulator environment
TEST_F(MeshDeviceSingleCardFixture, MultiDmAddTwoInts) {
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

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    KernelHandle kernel_0 = experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/add_two_ints.cpp",
        core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_processors_per_cluster = 5, .compile_args = {MEM_L1_UNCACHED_BASE}});

    KernelHandle kernel_1 = experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/add_two_ints.cpp",
        core,
        experimental::quasar::QuasarDataMovementConfig{
            .num_processors_per_cluster = 3, .compile_args = {MEM_L1_UNCACHED_BASE + sizeof(int)}});

    SetRuntimeArgs(program, kernel_0, core, {100, 200});
    SetRuntimeArgs(program, kernel_1, core, {300, 400});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> result{0, 0};
    tt_metal::detail::ReadFromDeviceL1(dev, core, 0, sizeof(int) * 2, result);

    ASSERT_EQ(result[0], 300) << "Got the value " << result[0] << " instead of " << 300;
    ASSERT_EQ(result[1], 700) << "Got the value " << result[1] << " instead of " << 700;
}
