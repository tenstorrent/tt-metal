// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Local invariants of KernelAdvancedOptions::global_semaphore_bindings.

#include <stdexcept>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalGen1ComputeKernel;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestGen1;

ProgramSpec MakeSpecWithGlobalSemaphoreParameterForTest(KernelSpec kernel) {
    ProgramSpec spec;
    spec.kernels = {std::move(kernel)};
    spec.work_units = {MakeMinimalWorkUnit("work", NodeCoord{0, 0}, {spec.kernels[0].unique_id.get()})};
    spec.advanced_options.global_semaphore_parameters = {{.unique_id = GlobalSemaphoreParamName{"sem"}}};
    return spec;
}

TEST_F(ProgramSpecTestGen1, CPU_GlobalSemaphoreBindingOnDMKernelIsValid) {
    // A data-movement kernel can bind a GlobalSemaphore.
    ProgramSpec spec = MakeSpecWithGlobalSemaphoreParameterForTest(MakeMinimalGen1DMKernel("writer"));
    spec.kernels[0].advanced_options.global_semaphore_bindings = {
        {.global_semaphore_parameter_name = GlobalSemaphoreParamName{"sem"}, .accessor_name = "signal"}};
    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestGen1, CPU_GlobalSemaphoreBindingOnComputeKernelFails) {
    // A compute kernel cannot bind a GlobalSemaphore.
    ProgramSpec spec = MakeSpecWithGlobalSemaphoreParameterForTest(MakeMinimalGen1ComputeKernel("compute"));
    spec.kernels[0].advanced_options.global_semaphore_bindings = {
        {.global_semaphore_parameter_name = GlobalSemaphoreParamName{"sem"}, .accessor_name = "signal"}};
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("compute kernel cannot bind")));
}

TEST_F(ProgramSpecTestGen1, CPU_GlobalSemaphoreBindingDuplicateAccessorNameFails) {
    // Two GlobalSemaphore bindings of one kernel cannot share an accessor name.
    ProgramSpec spec = MakeSpecWithGlobalSemaphoreParameterForTest(MakeMinimalGen1DMKernel("writer"));
    spec.advanced_options.global_semaphore_parameters.push_back({.unique_id = GlobalSemaphoreParamName{"sem2"}});
    spec.kernels[0].advanced_options.global_semaphore_bindings = {
        {.global_semaphore_parameter_name = GlobalSemaphoreParamName{"sem"}, .accessor_name = "signal"},
        {.global_semaphore_parameter_name = GlobalSemaphoreParamName{"sem2"}, .accessor_name = "signal"}};
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("duplicate GlobalSemaphore accessor_name 'signal'")));
}

TEST_F(ProgramSpecTestGen1, CPU_GlobalSemaphoreBindingInvalidAccessorNameFails) {
    // A GlobalSemaphore accessor name must be a valid C++ identifier.
    ProgramSpec spec = MakeSpecWithGlobalSemaphoreParameterForTest(MakeMinimalGen1DMKernel("writer"));
    spec.kernels[0].advanced_options.global_semaphore_bindings = {
        {.global_semaphore_parameter_name = GlobalSemaphoreParamName{"sem"}, .accessor_name = "not-an-identifier"}};
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("must be a valid C++ identifier")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
