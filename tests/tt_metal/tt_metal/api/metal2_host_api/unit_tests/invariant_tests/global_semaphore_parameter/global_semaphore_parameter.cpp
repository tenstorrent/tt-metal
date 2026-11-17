// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdexcept>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestGen1;

ProgramSpec MakeSpecWithOneDMKernelForTest() {
    ProgramSpec spec;
    spec.kernels = {MakeMinimalGen1DMKernel("writer")};
    spec.work_units = {MakeMinimalWorkUnit("work", NodeCoord{0, 0}, {"writer"})};
    return spec;
}

TEST_F(ProgramSpecTestGen1, CPU_GlobalSemaphoreParameterInL1SmallIsValid) {
    // A GlobalSemaphore parameter may live in L1_SMALL.
    ProgramSpec spec = MakeSpecWithOneDMKernelForTest();
    spec.advanced_options.global_semaphore_parameters = {
        {.unique_id = GlobalSemaphoreParamName{"sem"}, .memory_type = BufferType::L1_SMALL}};
    spec.kernels[0].advanced_options.global_semaphore_bindings = {
        {.global_semaphore_parameter_name = GlobalSemaphoreParamName{"sem"}, .accessor_name = "sem"}};
    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestGen1, CPU_GlobalSemaphoreParameterInDramFails) {
    // A GlobalSemaphore parameter cannot live in DRAM.
    ProgramSpec spec = MakeSpecWithOneDMKernelForTest();
    spec.advanced_options.global_semaphore_parameters = {
        {.unique_id = GlobalSemaphoreParamName{"sem"}, .memory_type = BufferType::DRAM}};
    spec.kernels[0].advanced_options.global_semaphore_bindings = {
        {.global_semaphore_parameter_name = GlobalSemaphoreParamName{"sem"}, .accessor_name = "sem"}};
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("must be L1 or L1_SMALL")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
