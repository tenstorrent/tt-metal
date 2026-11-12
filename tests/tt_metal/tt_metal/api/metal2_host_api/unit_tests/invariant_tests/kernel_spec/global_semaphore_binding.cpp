// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Invariants of ProgramSpec::global_semaphore_parameters and KernelSpec::global_semaphore_bindings:
// references, accessor names, kernel type, and every parameter being bound.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <stdexcept>
#include <string>
#include <vector>

#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::ProgramSpecTestGen1;
using test_helpers::ProgramSpecTestQuasar;

constexpr size_t kDmKernel = 0;
constexpr size_t kComputeKernel = 1;

GlobalSemaphoreBinding Bind(const std::string& param, const std::string& accessor) {
    return GlobalSemaphoreBinding{
        .global_semaphore_parameter_name = GlobalSemaphoreParamName{param}, .accessor_name = accessor};
}

ProgramSpec MakeSpecWithBinding() {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    spec.global_semaphore_parameters = {GlobalSemaphoreParameter{.unique_id = GlobalSemaphoreParamName{"sem"}}};
    spec.kernels[kDmKernel].global_semaphore_bindings = {Bind("sem", "sem")};
    return spec;
}

TEST_F(ProgramSpecTestGen1, CPU_GlobalSemaphoreBindingIsAccepted) {
    ProgramSpec spec = MakeSpecWithBinding();
    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestGen1, CPU_GlobalSemaphoreBoundByTwoKernelsIsAccepted) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto second_dm = test_helpers::MakeMinimalGen1DMKernel("dm_kernel_1", DataMovementProcessor::RISCV_1);
    spec.kernels.push_back(second_dm);
    spec.work_units = {test_helpers::MakeMinimalWorkUnit(
        "work_unit_0", NodeCoord{0, 0}, {"dm_kernel", "compute_kernel", "dm_kernel_1"})};
    spec.global_semaphore_parameters = {GlobalSemaphoreParameter{.unique_id = GlobalSemaphoreParamName{"sem"}}};
    spec.kernels[kDmKernel].global_semaphore_bindings = {Bind("sem", "a")};
    spec.kernels.back().global_semaphore_bindings = {Bind("sem", "b")};
    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestGen1, CPU_GlobalSemaphoreBindingToUnknownParameterFails) {
    ProgramSpec spec = MakeSpecWithBinding();
    spec.kernels[kDmKernel].global_semaphore_bindings = {Bind("sem", "sem"), Bind("missing", "other")};
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("references unknown GlobalSemaphoreParameter 'missing'")));
}

TEST_F(ProgramSpecTestGen1, CPU_UnboundGlobalSemaphoreParameterFails) {
    ProgramSpec spec = MakeSpecWithBinding();
    spec.global_semaphore_parameters.push_back(
        GlobalSemaphoreParameter{.unique_id = GlobalSemaphoreParamName{"unused"}});
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("GlobalSemaphoreParameter 'unused' is defined but not bound by any kernel")));
}

TEST_F(ProgramSpecTestGen1, CPU_DuplicateGlobalSemaphoreParameterFails) {
    ProgramSpec spec = MakeSpecWithBinding();
    spec.global_semaphore_parameters.push_back(GlobalSemaphoreParameter{.unique_id = GlobalSemaphoreParamName{"sem"}});
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Duplicate GlobalSemaphoreParameter")));
}

TEST_F(ProgramSpecTestGen1, CPU_GlobalSemaphoreBindingOnComputeKernelFails) {
    ProgramSpec spec = MakeSpecWithBinding();
    spec.kernels[kDmKernel].global_semaphore_bindings = {};
    spec.kernels[kComputeKernel].global_semaphore_bindings = {Bind("sem", "sem")};
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("only data-movement kernels can bind a GlobalSemaphore")));
}

TEST_F(ProgramSpecTestQuasar, CPU_GlobalSemaphoreBindingOnQuasarIsAccepted) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.global_semaphore_parameters = {GlobalSemaphoreParameter{.unique_id = GlobalSemaphoreParamName{"sem"}}};
    for (auto& kernel : spec.kernels) {
        if (kernel.is_data_movement_kernel()) {
            kernel.global_semaphore_bindings = {Bind("sem", "sem")};
            break;
        }
    }
    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestGen1, CPU_DuplicateGlobalSemaphoreAccessorNameFails) {
    ProgramSpec spec = MakeSpecWithBinding();
    spec.global_semaphore_parameters.push_back(
        GlobalSemaphoreParameter{.unique_id = GlobalSemaphoreParamName{"sem_1"}});
    spec.kernels[kDmKernel].global_semaphore_bindings = {Bind("sem", "dup"), Bind("sem_1", "dup")};
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("duplicate GlobalSemaphore accessor_name 'dup'")));
}

TEST_F(ProgramSpecTestGen1, CPU_InvalidGlobalSemaphoreAccessorNameFails) {
    for (const std::string bad_name : {"1bad", "has space"}) {
        ProgramSpec spec = MakeSpecWithBinding();
        spec.kernels[kDmKernel].global_semaphore_bindings = {Bind("sem", bad_name)};
        EXPECT_THAT(
            [&] { MakeProgramFromSpec(*mesh_device_, spec); },
            ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("must be a valid C++ identifier")))
            << "accessor_name: '" << bad_name << "'";
    }

    ProgramSpec spec = MakeSpecWithBinding();
    spec.kernels[kDmKernel].global_semaphore_bindings = {Bind("sem", std::string(MAX_ACCESSOR_NAME_LENGTH + 1, 'a'))};
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("must be at most")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
