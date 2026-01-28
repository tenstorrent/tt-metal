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
TEST_F(MeshDeviceSingleCardFixture, SingleDmRuntimeArgs) {
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

        constexpr uint32_t results_addr = MEM_L1_UNCACHED_BASE + (1000 * 1024);

        std::map<std::string, std::string> defines = {
            {"NUM_UNIQUE_RUNTIME_ARGS", std::to_string(num_runtime_args)},
            {"TEST_RUNTIME_ARGS", "1"},
            {"RESULTS_ADDR", std::to_string(results_addr)}};

        KernelHandle kernel = experimental::quasar::CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/misc/dm_configs_kernel.cpp",
            core,
            experimental::quasar::QuasarDataMovementConfig{.num_processors_per_cluster = 1, .defines = defines});

        SetRuntimeArgs(program, kernel, core, runtime_args);

        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, true);

        std::vector<uint32_t> result(num_runtime_args);
        tt_metal::detail::ReadFromDeviceL1(dev, core, results_addr, sizeof(uint32_t) * num_runtime_args, result);

        EXPECT_EQ(result, runtime_args);
    }
}

// This test requires simulator environment
TEST_F(MeshDeviceSingleCardFixture, SingleDmCommonRuntimeArgs) {
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

        constexpr uint32_t results_addr = MEM_L1_UNCACHED_BASE + (1000 * 1024);

        log_info(tt::LogMetal, "num_common_runtime_args = {}", num_common_runtime_args);

        std::map<std::string, std::string> defines = {
            {"NUM_COMMON_RUNTIME_ARGS", std::to_string(num_common_runtime_args)},
            {"TEST_RUNTIME_ARGS", "1"},
            {"RESULTS_ADDR", std::to_string(results_addr)}};

        KernelHandle kernel = experimental::quasar::CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/misc/dm_configs_kernel.cpp",
            core,
            experimental::quasar::QuasarDataMovementConfig{.num_processors_per_cluster = 1, .defines = defines});

        SetCommonRuntimeArgs(program, kernel, common_runtime_args);

        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, true);

        std::vector<uint32_t> result(num_common_runtime_args);
        tt_metal::detail::ReadFromDeviceL1(dev, core, results_addr, sizeof(uint32_t) * num_common_runtime_args, result);

        EXPECT_EQ(result, common_runtime_args);
    }
}

// This test requires simulator environment
TEST_F(MeshDeviceSingleCardFixture, SingleDmUniqueAndCommonRuntimeArgs) {
    // Skip if simulator is not available
    char* env_var = std::getenv("TT_METAL_SIMULATOR");
    if (env_var == nullptr) {
        GTEST_SKIP() << "This test can only be run using a simulator. Set TT_METAL_SIMULATOR environment variable.";
    }

    IDevice* dev = devices_[0]->get_devices()[0];
    auto mesh_device = devices_[0];
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

    constexpr uint32_t results_addr = MEM_L1_UNCACHED_BASE + (1000 * 1024);

    for (uint32_t num_common_runtime_args = 1; num_common_runtime_args <= max_runtime_args; num_common_runtime_args++) {
        for (uint32_t num_unique_runtime_args = 1; num_unique_runtime_args <= max_runtime_args;
             num_unique_runtime_args++) {
            if (num_unique_runtime_args + num_common_runtime_args > max_runtime_args) {
                continue;
            }

            distributed::MeshWorkload workload;
            distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
            Program program = CreateProgram();
            constexpr CoreCoord core = {0, 0};

            const uint32_t seed = std::time(nullptr);
            std::srand(seed);
            std::vector<uint32_t> unique_runtime_args(num_unique_runtime_args);
            for (uint32_t i = 0; i < num_unique_runtime_args; i++) {
                unique_runtime_args[i] = std::rand() % 1000;
            }
            std::vector<uint32_t> common_runtime_args(num_common_runtime_args);
            for (uint32_t i = 0; i < num_common_runtime_args; i++) {
                common_runtime_args[i] = std::rand() % 1000;
            }

            std::map<std::string, std::string> defines = {
                {"NUM_UNIQUE_RUNTIME_ARGS", std::to_string(num_unique_runtime_args)},
                {"NUM_COMMON_RUNTIME_ARGS", std::to_string(num_common_runtime_args)},
                {"TEST_RUNTIME_ARGS", "1"},
                {"RESULTS_ADDR", std::to_string(results_addr)}};
            KernelHandle kernel = experimental::quasar::CreateKernel(
                program,
                OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/misc/dm_configs_kernel.cpp",
                core,
                experimental::quasar::QuasarDataMovementConfig{.num_processors_per_cluster = 1, .defines = defines});

            SetRuntimeArgs(program, kernel, core, unique_runtime_args);
            SetCommonRuntimeArgs(program, kernel, common_runtime_args);

            workload.add_program(device_range, std::move(program));
            distributed::EnqueueMeshWorkload(cq, workload, true);

            std::vector<uint32_t> crta_result(num_common_runtime_args);
            tt_metal::detail::ReadFromDeviceL1(
                dev, core, results_addr, sizeof(uint32_t) * num_common_runtime_args, crta_result);

            std::vector<uint32_t> rta_result(num_unique_runtime_args);
            tt_metal::detail::ReadFromDeviceL1(
                dev,
                core,
                results_addr + (num_common_runtime_args * sizeof(uint32_t)),
                sizeof(uint32_t) * num_unique_runtime_args,
                rta_result);

            EXPECT_EQ(rta_result, unique_runtime_args);
            EXPECT_EQ(crta_result, common_runtime_args);
        }
    }
}

// This test requires simulator environment
TEST_F(MeshDeviceSingleCardFixture, SingleDmDefines) {
    // Skip if simulator is not available
    char* env_var = std::getenv("TT_METAL_SIMULATOR");
    if (env_var == nullptr) {
        GTEST_SKIP() << "This test can only be run using a simulator. Set TT_METAL_SIMULATOR environment variable.";
    }

    IDevice* dev = devices_[0]->get_devices()[0];
    auto mesh_device = devices_[0];
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

    constexpr uint32_t results_addr = MEM_L1_UNCACHED_BASE + (1000 * 1024);

    for (uint32_t i = 0; i < 100; i++) {
        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
        Program program = CreateProgram();
        constexpr CoreCoord core = {0, 0};

        const uint32_t seed = std::time(nullptr);
        std::srand(seed);
        std::map<std::string, std::string> defines = {
            {"DEFINES_0", std::to_string(std::rand() % 1000)},
            {"DEFINES_1", std::to_string(std::rand() % 1000)},
            {"DEFINES_2", std::to_string(std::rand() % 1000)},
            {"TEST_DEFINES", "1"},
            {"RESULTS_ADDR", std::to_string(results_addr)}};
        experimental::quasar::CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/misc/dm_configs_kernel.cpp",
            core,
            experimental::quasar::QuasarDataMovementConfig{.num_processors_per_cluster = 1, .defines = defines});

        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, true);

        std::vector<uint32_t> result(3);
        tt_metal::detail::ReadFromDeviceL1(dev, core, results_addr, sizeof(uint32_t) * 3, result);

        EXPECT_EQ(result[0], std::stoi(defines["DEFINES_0"]));
        EXPECT_EQ(result[1], std::stoi(defines["DEFINES_1"]));
        EXPECT_EQ(result[2], std::stoi(defines["DEFINES_2"]));
    }
}

// This test requires simulator environment
TEST_F(MeshDeviceSingleCardFixture, SingleDmCompileArgs) {
    // Skip if simulator is not available
    char* env_var = std::getenv("TT_METAL_SIMULATOR");
    if (env_var == nullptr) {
        GTEST_SKIP() << "This test can only be run using a simulator. Set TT_METAL_SIMULATOR environment variable.";
    }

    IDevice* dev = devices_[0]->get_devices()[0];
    auto mesh_device = devices_[0];
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

    constexpr uint32_t results_addr = MEM_L1_UNCACHED_BASE + (1000 * 1024);

    for (uint32_t i = 0; i < 100; i++) {
        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
        Program program = CreateProgram();
        constexpr CoreCoord core = {0, 0};

        const uint32_t seed = std::time(nullptr);
        std::srand(seed);
        std::map<std::string, std::string> defines = {
            {"TEST_COMPILE_ARGS", "1"}, {"RESULTS_ADDR", std::to_string(results_addr)}};
        std::vector<uint32_t> compile_args(5);
        for (uint32_t j = 0; j < 5; j++) {
            compile_args[j] = std::rand() % 1000;
        }
        experimental::quasar::CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/misc/dm_configs_kernel.cpp",
            core,
            experimental::quasar::QuasarDataMovementConfig{
                .num_processors_per_cluster = 1, .compile_args = compile_args, .defines = defines});

        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, true);

        std::vector<uint32_t> result(5);
        tt_metal::detail::ReadFromDeviceL1(dev, core, results_addr, sizeof(uint32_t) * 5, result);

        EXPECT_EQ(result, compile_args);
    }
}

// This test requires simulator environment
TEST_F(MeshDeviceSingleCardFixture, SingleDmNamedCompileArgs) {
    // Skip if simulator is not available
    char* env_var = std::getenv("TT_METAL_SIMULATOR");
    if (env_var == nullptr) {
        GTEST_SKIP() << "This test can only be run using a simulator. Set TT_METAL_SIMULATOR environment variable.";
    }

    IDevice* dev = devices_[0]->get_devices()[0];
    auto mesh_device = devices_[0];
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

    constexpr uint32_t results_addr = MEM_L1_UNCACHED_BASE + (1000 * 1024);

    for (uint32_t i = 0; i < 100; i++) {
        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
        Program program = CreateProgram();
        constexpr CoreCoord core = {0, 0};

        const uint32_t seed = std::time(nullptr);
        std::srand(seed);
        std::map<std::string, std::string> defines = {
            {"TEST_NAMED_COMPILE_ARGS", "1"}, {"RESULTS_ADDR", std::to_string(results_addr)}};
        std::unordered_map<std::string, uint32_t> named_compile_args = {
            {"NAMED_COMPILE_ARGS_0", std::rand() % 1000},
            {"NAMED_COMPILE_ARGS_1", std::rand() % 1000},
            {"NAMED_COMPILE_ARGS_2", std::rand() % 1000},
            {"NAMED_COMPILE_ARGS_3", std::rand() % 1000},
            {"NAMED_COMPILE_ARGS_4", std::rand() % 1000}};

        experimental::quasar::CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/misc/dm_configs_kernel.cpp",
            core,
            experimental::quasar::QuasarDataMovementConfig{
                .num_processors_per_cluster = 1, .defines = defines, .named_compile_args = named_compile_args});

        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, true);

        std::vector<uint32_t> result(5);
        tt_metal::detail::ReadFromDeviceL1(dev, core, results_addr, sizeof(uint32_t) * 5, result);

        EXPECT_EQ(result[0], named_compile_args["NAMED_COMPILE_ARGS_0"]);
        EXPECT_EQ(result[1], named_compile_args["NAMED_COMPILE_ARGS_1"]);
        EXPECT_EQ(result[2], named_compile_args["NAMED_COMPILE_ARGS_2"]);
        EXPECT_EQ(result[3], named_compile_args["NAMED_COMPILE_ARGS_3"]);
        EXPECT_EQ(result[4], named_compile_args["NAMED_COMPILE_ARGS_4"]);
    }
}
