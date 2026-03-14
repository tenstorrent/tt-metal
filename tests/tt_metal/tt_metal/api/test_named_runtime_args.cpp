// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tests for named args (CT and RT) with generated header.
//
// Verifies:
// 1. Named common runtime args delivered via rt_args::get<>()
// 2. Named per-core runtime args deliver different values per core
// 3. Named compile-time args accessible via ct_args:: namespace

#include <gtest/gtest.h>
#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "device_fixture.hpp"
#include "multi_device_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;
using NamedArgsTest = GenericMeshDeviceFixture;

TEST_F(NamedArgsTest, TensixTestNamedCommonAndPerCoreRuntimeArgs) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->get_devices()[0];
    auto& cq = mesh_device->mesh_command_queue();
    auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    CoreCoord core0 = {0, 0};
    CoreCoord core1 = {1, 0};
    CoreRangeSet cores = std::set<CoreRange>({CoreRange(core0, core1)});

    const uint32_t write_addr = mesh_device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);

    const uint32_t expected_marker = 0xCAFE;
    const uint32_t core0_idx = 10;
    const uint32_t core1_idx = 20;

    KernelDescriptor kernel = {
        .kernel_source = "tests/tt_metal/tt_metal/test_kernels/misc/named_runtime_args_kernel.cpp",
        .core_ranges = cores,
        .named_compile_time_args = {{"my_kernel.param_a", 0}, {"my_kernel.param_b", 0}},
        .defines = {{"WRITE_ADDRESS", std::to_string(write_addr)}},
        .named_common_runtime_args = {{"my_kernel.marker", expected_marker}},
        .named_per_core_runtime_args = {{"my_kernel.core_idx", {{core0, core0_idx}, {core1, core1_idx}}}},
        .config = DataMovementConfigDescriptor{},
    };

    distributed::MeshWorkload workload;
    Program program(ProgramDescriptor{.kernels = {kernel}});
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);

    std::vector<uint32_t> results_core0;
    detail::ReadFromDeviceL1(device, core0, write_addr, 2 * sizeof(uint32_t), results_core0);
    std::vector<uint32_t> results_core1;
    detail::ReadFromDeviceL1(device, core1, write_addr, 2 * sizeof(uint32_t), results_core1);

    EXPECT_EQ(results_core0[0], expected_marker) << "Core (0,0): marker should be 0xCAFE";
    EXPECT_EQ(results_core1[0], expected_marker) << "Core (1,0): marker should be 0xCAFE";
    EXPECT_EQ(results_core0[1], core0_idx) << "Core (0,0): core_idx should be 10";
    EXPECT_EQ(results_core1[1], core1_idx) << "Core (1,0): core_idx should be 20";
}

TEST_F(NamedArgsTest, TensixTestNamedCompileTimeArgs) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->get_devices()[0];
    auto& cq = mesh_device->mesh_command_queue();
    auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    CoreCoord core = {0, 0};
    CoreRangeSet cores = std::set<CoreRange>({CoreRange(core, core)});

    const uint32_t write_addr = mesh_device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);

    const uint32_t param_a = 42;
    const uint32_t param_b = 0xBEEF;

    KernelDescriptor kernel = {
        .kernel_source = "tests/tt_metal/tt_metal/test_kernels/misc/named_runtime_args_kernel.cpp",
        .core_ranges = cores,
        .named_compile_time_args = {{"my_kernel.param_a", param_a}, {"my_kernel.param_b", param_b}},
        .defines = {{"WRITE_ADDRESS", std::to_string(write_addr)}},
        .named_common_runtime_args = {{"my_kernel.marker", 0}},
        .named_per_core_runtime_args = {{"my_kernel.core_idx", {{core, 0}}}},
        .config = DataMovementConfigDescriptor{},
    };

    distributed::MeshWorkload workload;
    Program program(ProgramDescriptor{.kernels = {kernel}});
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);

    std::vector<uint32_t> results;
    detail::ReadFromDeviceL1(device, core, write_addr, 4 * sizeof(uint32_t), results);

    EXPECT_EQ(results[2], param_a) << "ct_args::my_kernel::param_a should be 42";
    EXPECT_EQ(results[3], param_b) << "ct_args::my_kernel::param_b should be 0xBEEF";
}
