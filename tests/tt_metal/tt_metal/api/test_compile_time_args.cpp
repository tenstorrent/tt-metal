// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <stdint.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <map>
#include <unordered_map>
#include <numeric>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "device_fixture.hpp"
#include "multi_device_fixture.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/utils.hpp>

namespace tt {
namespace tt_metal {
class IDevice;
}  // namespace tt_metal
}  // namespace tt

using namespace tt;
using namespace tt::tt_metal;
using CompileTimeArgsTest = GenericMeshDeviceFixture;

TEST_F(MeshDeviceFixture, TensixTestTwentyThousandCompileTimeArgs) {
    for (const auto& mesh_device : this->devices_) {
        CoreCoord core = {0, 0};
        auto& cq = mesh_device->mesh_command_queue();
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord);
        distributed::MeshWorkload workload;
        Program program;
        distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);
        auto& program_ = workload.get_programs().at(device_range);
        auto device = mesh_device->get_devices()[0];

        const uint32_t write_addr = mesh_device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);

        const std::map<std::string, std::string>& defines = {{"WRITE_ADDRESS", std::to_string(write_addr)}};

        const uint32_t num_compile_time_args = 20000;
        std::vector<uint32_t> compile_time_args(num_compile_time_args);
        std::iota(compile_time_args.begin(), compile_time_args.end(), 0);

        CreateKernel(
            program_,
            "tests/tt_metal/tt_metal/test_kernels/misc/compile_time_args_kernel.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = compile_time_args,
                .defines = defines});
        distributed::EnqueueMeshWorkload(cq, workload, false);

        const std::vector<uint32_t> compile_time_args_expected{
            std::accumulate(compile_time_args.begin(), compile_time_args.end(), 0u)};

        std::vector<uint32_t> compile_time_args_actual;
        detail::ReadFromDeviceL1(device, core, write_addr, sizeof(uint32_t), compile_time_args_actual);

        ASSERT_EQ(compile_time_args_actual, compile_time_args_expected);
    }
}

TEST_F(CompileTimeArgsTest, TensixTestNamedCompileTimeArgs) {
    auto mesh_device = get_mesh_device();
    CoreCoord core = {0, 0};
    auto& cq = mesh_device->mesh_command_queue();
    auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    distributed::MeshWorkload workload;
    Program program;
    distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);
    auto& program_ = workload.get_programs().at(device_range);
    auto device = mesh_device->get_devices()[0];

    const uint32_t write_addr = mesh_device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);

    const std::map<std::string, std::string> defines = {{"WRITE_ADDRESS", std::to_string(write_addr)}};

    const std::vector<uint32_t> compile_time_args = {12, 456, 1024, 3};
    const std::unordered_map<std::string, uint32_t> named_compile_time_args = {
        {"buffer_size", 1024},
        {"", 3},
        {"!@#$%^&*()", 12},
        {"very_long_parameter_name_that_someone_could_potentially_use_to_try_to_break_the_kernel", 456}};

    CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/misc/named_compile_time_args_kernel.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = compile_time_args,
            .defines = defines,
            .named_compile_args = named_compile_time_args});
    distributed::EnqueueMeshWorkload(cq, workload, false);

    std::vector<uint32_t> results;
    detail::ReadFromDeviceL1(device, core, write_addr, 4 * sizeof(uint32_t), results);

    ASSERT_EQ(results[0], compile_time_args[0]) << "'!@#$%^&*()' should be 12";
    ASSERT_EQ(results[1], compile_time_args[1])
        << "'very_long_parameter_name_that_someone_could_potentially_use_to_try_to_break_the_kernel' should be 456";
    ASSERT_EQ(results[2], compile_time_args[2]) << "'buffer_size' should be 1024";
    ASSERT_EQ(results[3], compile_time_args[3]) << "\"\" should be 3";
}
