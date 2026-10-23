// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Local invariants of KernelSpec::SemaphoreBinding and KernelSpec::semaphore_bindings (kernel_spec.hpp):
// accessor names, and which architectures allow semaphores on compute kernels.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <stdexcept>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::ProgramSpecTestGen1;
using test_helpers::ProgramSpecTestQuasar;

TEST_F(ProgramSpecTestQuasar, CPU_KernelSemaphoreBindingsSucceed) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    SemaphoreSpec sem;
    sem.unique_id = SemaphoreSpecName{"sem_0"};
    sem.target_nodes = NodeCoord{0, 0};
    spec.semaphores = {sem};

    SemaphoreBinding binding;
    binding.semaphore_spec_name = SemaphoreSpecName{"sem_0"};
    binding.accessor_name = "my_sem";
    spec.kernels[0].semaphore_bindings = {binding};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_KernelSemaphoreBindingInvalidAccessorFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    SemaphoreSpec sem;
    sem.unique_id = SemaphoreSpecName{"sem_0"};
    sem.target_nodes = NodeCoord{0, 0};
    spec.semaphores = {sem};

    SemaphoreBinding binding;
    binding.semaphore_spec_name = SemaphoreSpecName{"sem_0"};
    binding.accessor_name = "has-dash";
    spec.kernels[0].semaphore_bindings = {binding};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("semaphore accessor_name 'has-dash' must be a valid C++ identifier")));
}

TEST_F(ProgramSpecTestQuasar, CPU_KernelSemaphoreBindingDuplicateAccessorFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    SemaphoreSpec sem0;
    sem0.unique_id = SemaphoreSpecName{"sem_0"};
    sem0.target_nodes = NodeCoord{0, 0};

    SemaphoreSpec sem1;
    sem1.unique_id = SemaphoreSpecName{"sem_1"};
    sem1.target_nodes = NodeCoord{0, 0};

    spec.semaphores = {sem0, sem1};

    spec.kernels[0].semaphore_bindings = {
        SemaphoreBinding{.semaphore_spec_name = SemaphoreSpecName{"sem_0"}, .accessor_name = "same"},
        SemaphoreBinding{.semaphore_spec_name = SemaphoreSpecName{"sem_1"}, .accessor_name = "same"}};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("duplicate semaphore accessor_name 'same'")));
}

TEST_F(ProgramSpecTestQuasar, CPU_SemaphoreBoundToComputeKernelFailsOnQuasar) {
    // Compute kernels cannot have semaphore bindings on any arch.
    // (This may later change for Quasar.)
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    SemaphoreSpec sem;
    sem.unique_id = SemaphoreSpecName{"sem_0"};
    sem.target_nodes = NodeCoord{0, 0};
    spec.semaphores = {sem};

    // kernels[1] is the compute kernel in MakeMinimalValidProgramSpec
    ASSERT_TRUE(spec.kernels[1].is_compute_kernel());
    spec.kernels[1].semaphore_bindings = {
        SemaphoreBinding{.semaphore_spec_name = SemaphoreSpecName{"sem_0"}, .accessor_name = "done_flag"}};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Semaphore bindings on compute kernels are supported only on Blackhole.")));
}

TEST_F(ProgramSpecTestGen1, CPU_SemaphoreBoundToComputeKernelFailsOnWormhole) {
    // Wormhole compute kernels cannot have semaphore bindings
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    SemaphoreSpec sem;
    sem.unique_id = SemaphoreSpecName{"sem_0"};
    sem.target_nodes = NodeCoord{0, 0};
    spec.semaphores = {sem};

    // kernels[1] is the compute kernel in MakeMinimalGen1ValidProgramSpec
    ASSERT_TRUE(spec.kernels[1].is_compute_kernel());
    spec.kernels[1].semaphore_bindings = {
        SemaphoreBinding{.semaphore_spec_name = SemaphoreSpecName{"sem_0"}, .accessor_name = "done_flag"}};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Semaphore bindings on compute kernels are supported only on Blackhole.")));
}

TEST_F(ProgramSpecTestGen1, CPU_SemaphoreBoundToDMKernelSucceedsOnGen1) {
    // Sanity check: binding a semaphore to a DM kernel on WH/BH is allowed.
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    SemaphoreSpec sem;
    sem.unique_id = SemaphoreSpecName{"sem_0"};
    sem.target_nodes = NodeCoord{0, 0};
    spec.semaphores = {sem};

    // kernels[0] is the DM kernel in MakeMinimalGen1ValidProgramSpec
    ASSERT_TRUE(spec.kernels[0].is_data_movement_kernel());
    spec.kernels[0].semaphore_bindings = {
        SemaphoreBinding{.semaphore_spec_name = SemaphoreSpecName{"sem_0"}, .accessor_name = "done_flag"}};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
