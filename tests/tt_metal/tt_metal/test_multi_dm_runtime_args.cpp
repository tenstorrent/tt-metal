// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "common/device_fixture.hpp"

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "hw/inc/internal/tt-2xx/quasar/dev_mem_map.h"
#include "kernel_types.hpp"

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

// This test requires simulator environment
TEST_F(MeshDeviceSingleCardFixture, MultiDmRuntimeArgs) {
    // Skip if simulator is not available
    char* env_var = std::getenv("TT_METAL_SIMULATOR");
    if (env_var == nullptr) {
        GTEST_SKIP() << "This test can only be run using a simulator. Set TT_METAL_SIMULATOR environment variable.";
    }

    IDevice* dev = devices_[0]->get_devices()[0];
    auto mesh_device = devices_[0];

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    for (uint32_t num_runtime_args = 1; num_runtime_args <= max_runtime_args; num_runtime_args++) {
        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
        Program program = CreateProgram();
        constexpr CoreCoord core = {0, 0};

        const uint32_t seed = std::time(nullptr);
        std::srand(seed);
        std::vector<uint32_t> runtime_args(num_runtime_args);
        for (uint32_t i = 0; i < num_runtime_args; i++) {
            runtime_args[i] = std::rand() % 1000;
        }

        std::map<std::string, std::string> defines = {
            {"DATA_MOVEMENT", "1"},
            {"NUM_RUNTIME_ARGS", std::to_string(num_runtime_args)},
            {"RESULTS_ADDR", std::to_string(MEM_L1_UNCACHED_BASE)}};

        KernelHandle kernel = experimental::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/runtime_args_kernel.cpp",
            core,
            experimental::QuasarDataMovementConfig{.num_processors_per_cluster = 8, .defines = defines});

        SetRuntimeArgs(program, kernel, core, runtime_args);

        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, true);

        std::vector<uint32_t> result(num_runtime_args);
        tt_metal::detail::ReadFromDeviceL1(
            dev, core, MEM_L1_UNCACHED_BASE, sizeof(uint32_t) * num_runtime_args, result);

        EXPECT_EQ(result, runtime_args);
    }
}

// This test requires simulator environment
TEST_F(MeshDeviceSingleCardFixture, MultiDmCommonRuntimeArgs) {
    // Skip if simulator is not available
    char* env_var = std::getenv("TT_METAL_SIMULATOR");
    if (env_var == nullptr) {
        GTEST_SKIP() << "This test can only be run using a simulator. Set TT_METAL_SIMULATOR environment variable.";
    }

    IDevice* dev = devices_[0]->get_devices()[0];
    auto mesh_device = devices_[0];

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    for (uint32_t num_common_runtime_args = 1; num_common_runtime_args <= max_runtime_args; num_common_runtime_args++) {
        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
        Program program = CreateProgram();
        constexpr CoreCoord core = {0, 0};

        const uint32_t seed = std::time(nullptr);
        std::srand(seed);
        std::vector<uint32_t> common_runtime_args(num_common_runtime_args);
        for (uint32_t i = 0; i < num_common_runtime_args; i++) {
            common_runtime_args[i] = std::rand() % 1000;
        }

        std::map<std::string, std::string> defines = {
            {"DATA_MOVEMENT", "1"},
            {"NUM_RUNTIME_ARGS", std::to_string(num_common_runtime_args)},
            {"RESULTS_ADDR", std::to_string(MEM_L1_UNCACHED_BASE)},
            {"COMMON_RUNTIME_ARGS", "1"}};

        KernelHandle kernel = experimental::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/runtime_args_kernel.cpp",
            core,
            experimental::QuasarDataMovementConfig{.num_processors_per_cluster = 8, .defines = defines});

        SetCommonRuntimeArgs(program, kernel, common_runtime_args);

        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, true);

        std::vector<uint32_t> result(num_common_runtime_args);
        tt_metal::detail::ReadFromDeviceL1(
            dev, core, MEM_L1_UNCACHED_BASE, sizeof(uint32_t) * num_common_runtime_args, result);

        EXPECT_EQ(result, common_runtime_args);
    }
}
