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
        std::cerr
            << "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to (0,0),(1,0) to see the output of "
               "the Data Movement kernels."
            << std::endl;
        std::cerr << "WARNING: For example, export TT_METAL_DPRINT_CORES=(0,0),(1,0)" << std::endl;
    }

    IDevice* dev = devices_[0]->get_devices()[0];
    auto mesh_device = devices_[0];

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();
    const CoreRange core_range = {{0, 0}, {1, 0}};

    KernelHandle kernel_0 = experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/add_two_ints.cpp",
        core_range,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 4, .compile_args = {MEM_L1_UNCACHED_BASE}});

    KernelHandle kernel_1 = experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/add_two_ints.cpp",
        CoreCoord(0, 0),
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 3, .compile_args = {MEM_L1_UNCACHED_BASE + sizeof(int)}});

    KernelHandle kernel_2 = experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/add_two_ints.cpp",
        CoreCoord(1, 0),
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 4, .compile_args = {MEM_L1_UNCACHED_BASE + sizeof(int)}});

    SetRuntimeArgs(program, kernel_0, core_range, {1, 2});
    SetRuntimeArgs(program, kernel_1, CoreCoord(0, 0), {3, 4});
    SetRuntimeArgs(program, kernel_2, CoreCoord(1, 0), {5, 6});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    std::vector<uint32_t> result_core_0(2, 0);
    tt_metal::detail::ReadFromDeviceL1(dev, CoreCoord(0, 0), 0, sizeof(uint32_t) * 2, result_core_0);

    std::vector<uint32_t> result_core_1(2, 0);
    tt_metal::detail::ReadFromDeviceL1(dev, CoreCoord(1, 0), 0, sizeof(uint32_t) * 2, result_core_1);

    ASSERT_EQ(result_core_0, (std::vector<uint32_t>{3, 7}));
    ASSERT_EQ(result_core_1, (std::vector<uint32_t>{3, 11}));
}
