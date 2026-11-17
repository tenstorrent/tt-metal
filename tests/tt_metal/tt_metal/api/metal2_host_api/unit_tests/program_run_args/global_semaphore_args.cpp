// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// GlobalSemaphore arguments (AdvancedProgramRunArgs::global_semaphore_args).

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramRunArgsTestGen1;

// A data-movement kernel "writer" on node {0, 0} binds GlobalSemaphore parameter "sem" as "signal".
ProgramSpec MakeSpecWithGlobalSemaphoreBindingForTest(BufferType memory_type = BufferType::L1) {
    ProgramSpec spec;
    spec.kernels = {MakeMinimalGen1DMKernel("writer")};
    spec.kernels[0].advanced_options.global_semaphore_bindings = {
        {.global_semaphore_parameter_name = GlobalSemaphoreParamName{"sem"}, .accessor_name = "signal"}};
    spec.advanced_options.global_semaphore_parameters = {
        {.unique_id = GlobalSemaphoreParamName{"sem"}, .memory_type = memory_type}};
    spec.work_units = {MakeMinimalWorkUnit("work", NodeCoord{0, 0}, {"writer"})};
    return spec;
}

ProgramRunArgs GlobalSemaphoreArgs(const GlobalSemaphore& semaphore, const char* name = "sem") {
    ProgramRunArgs args;
    args.advanced_options.global_semaphore_args = {{GlobalSemaphoreParamName{name}, std::cref(semaphore)}};
    return args;
}

// Not a public API: the kernel's common runtime arg buffer, as the device will read it.
std::vector<uint32_t> CommonRuntimeArgsOfKernel(Program& program, const std::string& kernel_name) {
    auto kernel = program.impl().get_kernel_by_spec_name(kernel_name);
    const RuntimeArgsData& crtas = kernel->common_runtime_args_data();
    return {crtas.data(), crtas.data() + crtas.size()};
}

TEST_F(ProgramRunArgsTestGen1, CPU_GlobalSemaphoreArgumentMissingFails) {
    // SetProgramRunArgs requires a GlobalSemaphore for every declared parameter.
    Program program = MakeProgramFromSpec(*mesh_device_, MakeSpecWithGlobalSemaphoreBindingForTest());
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, {}); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("has no GlobalSemaphoreArgument")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_GlobalSemaphoreArgumentUnknownParameterFails) {
    // A GlobalSemaphore argument must name a declared parameter.
    GlobalSemaphore semaphore(*mesh_device_, CoreRangeSet(CoreRange({0, 0})), 0);
    Program program = MakeProgramFromSpec(*mesh_device_, MakeSpecWithGlobalSemaphoreBindingForTest());
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, GlobalSemaphoreArgs(semaphore, "unknown")); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("unknown GlobalSemaphoreParameter")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_GlobalSemaphoreArgumentMemoryTypeMismatchFails) {
    // A GlobalSemaphore must live in the memory type its parameter declares.
    GlobalSemaphore semaphore(*mesh_device_, CoreRangeSet(CoreRange({0, 0})), 0, BufferType::L1);
    Program program =
        MakeProgramFromSpec(*mesh_device_, MakeSpecWithGlobalSemaphoreBindingForTest(BufferType::L1_SMALL));
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, GlobalSemaphoreArgs(semaphore)); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("memory type")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_GlobalSemaphoreArgumentNotOnKernelNodeFails) {
    // A GlobalSemaphore must span every node of every kernel that binds its parameter.
    GlobalSemaphore semaphore(*mesh_device_, CoreRangeSet(CoreRange({1, 0})), 0);
    Program program = MakeProgramFromSpec(*mesh_device_, MakeSpecWithGlobalSemaphoreBindingForTest());
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, GlobalSemaphoreArgs(semaphore)); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("does not span node")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_GlobalSemaphoreAddressInCommonRuntimeArgs) {
    // The bound GlobalSemaphore's address reaches the kernel through its common runtime args.
    GlobalSemaphore semaphore(*mesh_device_, CoreRangeSet(CoreRange({0, 0})), 0);
    Program program = MakeProgramFromSpec(*mesh_device_, MakeSpecWithGlobalSemaphoreBindingForTest());
    SetProgramRunArgs(program, GlobalSemaphoreArgs(semaphore));
    EXPECT_THAT(
        CommonRuntimeArgsOfKernel(program, "writer"),
        ::testing::ElementsAre(static_cast<uint32_t>(semaphore.address())));
}

TEST_F(ProgramRunArgsTestGen1, CPU_UpdateProgramRunArgsReplacesGlobalSemaphore) {
    // UpdateProgramRunArgs can bind a different GlobalSemaphore.
    GlobalSemaphore first(*mesh_device_, CoreRangeSet(CoreRange({0, 0})), 0);
    GlobalSemaphore second(*mesh_device_, CoreRangeSet(CoreRange({0, 0})), 0);
    Program program = MakeProgramFromSpec(*mesh_device_, MakeSpecWithGlobalSemaphoreBindingForTest());
    SetProgramRunArgs(program, GlobalSemaphoreArgs(first));
    UpdateProgramRunArgs(program, GlobalSemaphoreArgs(second));
    EXPECT_THAT(
        CommonRuntimeArgsOfKernel(program, "writer"), ::testing::ElementsAre(static_cast<uint32_t>(second.address())));
}

TEST_F(ProgramRunArgsTestGen1, CPU_UpdateProgramRunArgsKeepsOmittedGlobalSemaphore) {
    // UpdateProgramRunArgs keeps the GlobalSemaphore of a parameter it omits.
    GlobalSemaphore semaphore(*mesh_device_, CoreRangeSet(CoreRange({0, 0})), 0);
    Program program = MakeProgramFromSpec(*mesh_device_, MakeSpecWithGlobalSemaphoreBindingForTest());
    SetProgramRunArgs(program, GlobalSemaphoreArgs(semaphore));
    UpdateProgramRunArgs(program, {});
    EXPECT_THAT(
        CommonRuntimeArgsOfKernel(program, "writer"),
        ::testing::ElementsAre(static_cast<uint32_t>(semaphore.address())));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
