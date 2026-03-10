// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "hw/inc/internal/tt-2xx/quasar/dev_mem_map.h"
#include "llrt/rtoptions.hpp"

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

// This test requires simulator environment
TEST_F(MeshDeviceSingleCardFixture, MultiDmAddTwoInts) {
    // Skip if simulator is not available
    if (!MetalContext::instance().rtoptions().get_simulator_enabled()) {
        GTEST_SKIP() << "This test can only be run using a simulator. Set TT_METAL_SIMULATOR environment variable.";
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
    Program program = CreateProgram();
    const CoreRange core_range = {{0, 0}};

    KernelHandle kernel_0 = experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/add_two_ints.cpp",
        core_range,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 3, .compile_args = {MEM_L1_UNCACHED_BASE}});

    KernelHandle kernel_1 = experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/add_two_ints.cpp",
        core_range,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 2, .compile_args = {MEM_L1_UNCACHED_BASE + sizeof(int)}});

    KernelHandle kernel_2 = experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/add_two_ints.cpp",
        CoreCoord(0, 0),
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 3, .compile_args = {MEM_L1_UNCACHED_BASE + (2 * sizeof(int))}});

    // KernelHandle kernel_3 = experimental::quasar::CreateKernel(
    //     program,
    //     "tests/tt_metal/tt_metal/test_kernels/misc/add_two_ints.cpp",
    //     CoreCoord(1, 0),
    //     experimental::quasar::QuasarDataMovementConfig{
    //         .num_threads_per_cluster = 2, .compile_args = {MEM_L1_UNCACHED_BASE + (2 * sizeof(int))}});

    SetRuntimeArgs(program, kernel_0, core_range, {1, 2});
    SetRuntimeArgs(program, kernel_1, core_range, {3, 4});
    SetRuntimeArgs(program, kernel_2, CoreCoord(0, 0), {5, 6});
    // SetRuntimeArgs(program, kernel_3, CoreCoord(1, 0), {7, 8});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> result_core_0(3, 0);
    tt_metal::detail::ReadFromDeviceL1(dev, CoreCoord(0, 0), 0, sizeof(uint32_t) * 3, result_core_0);

    // std::vector<uint32_t> result_core_1(3, 0);
    // tt_metal::detail::ReadFromDeviceL1(dev, CoreCoord(1, 0), 0, sizeof(uint32_t) * 3, result_core_1);

    ASSERT_EQ(result_core_0, (std::vector<uint32_t>{3, 7, 11}));
    // ASSERT_EQ(result_core_1, (std::vector<uint32_t>{3, 7, 15}));
}
