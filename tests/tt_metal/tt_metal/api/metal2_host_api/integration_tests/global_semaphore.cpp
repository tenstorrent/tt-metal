// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// GlobalSemaphore bindings on Gen1 silicon (slow dispatch).

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <cstdint>
#include <functional>
#include <vector>

#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "impl/program/program_impl.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include "metal2_host_api/test_helpers/program_spec_hw_fixture.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecHWTest;

// A kernel on `node` that sets GlobalSemaphore parameter "sem" to its named RTA "value".
ProgramSpec MakeGlobalSemaphoreSetSpecForTest(const NodeCoord& node) {
    KernelSpec setter = MakeMinimalGen1DMKernel("setter");
    setter.source = "tests/tt_metal/tt_metal/test_kernels/dataflow/global_semaphore_set.cpp";
    setter.runtime_arg_schema.runtime_arg_names = {"value"};
    setter.advanced_options.global_semaphore_bindings = {
        {.global_semaphore_parameter_name = GlobalSemaphoreParamName{"sem"}, .accessor_name = "signal"}};
    ProgramSpec spec{.kernels = {setter}, .work_units = {MakeMinimalWorkUnit("work", node, {"setter"})}};
    spec.advanced_options.global_semaphore_parameters = {{.unique_id = GlobalSemaphoreParamName{"sem"}}};
    return spec;
}

TEST_F(ProgramSpecHWTest, GlobalSemaphoreSetByKernelIsVisibleAtItsAddress) {
    // A kernel writes the GlobalSemaphore that the run args bind, at the GlobalSemaphore's address.
    auto mesh_device = devices_.at(0);
    const NodeCoord node{0, 0};
    GlobalSemaphore semaphore(*mesh_device, CoreRangeSet(CoreRange(node)), 0);
    Program program = MakeProgramFromSpec(*mesh_device, MakeGlobalSemaphoreSetSpecForTest(node));
    ProgramRunArgs args;
    args.kernel_run_args = {
        {.kernel = KernelSpecName{"setter"},
         .runtime_arg_values = MakeRuntimeArgsForSingleNode(node, {{"value", 0xC0DEu}})}};
    args.advanced_options.global_semaphore_args = {{GlobalSemaphoreParamName{"sem"}, std::cref(semaphore)}};
    SetProgramRunArgs(program, args);
    LaunchProgram(*mesh_device, std::move(program));
    std::vector<uint32_t> semaphore_value;
    const auto address = static_cast<uint32_t>(semaphore.address());
    slow_dispatch::ReadFromL1(*mesh_device, node, address, sizeof(uint32_t), semaphore_value);
    EXPECT_THAT(semaphore_value, ::testing::ElementsAre(0xC0DEu));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
