// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ProgramSpec structural invariants on semaphores (program_spec.hpp): compute-bound semaphores are not
// shared with DM kernels, at most one exists, and their advanced options are constrained.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <stdexcept>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindSemaphoreToKernels;
using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::ProgramSpecTestBlackhole;

// A compute semaphore synchronizes UNPACK with PACK and may not be shared with a DM kernel: it is
// a Tensix hardware (Sync Unit) semaphore that a DM core cannot reach, so there is no scope that
// can serve both binders. Rejected at validation rather than resolved into a mechanism only one
// side can drive.
TEST_F(ProgramSpecTestBlackhole, CPU_SemaphoreSharedByComputeAndDMIsRejected) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    BindSemaphoreToKernels(spec, "shared_sem", {"dm_kernel", "compute_kernel"});

    EXPECT_ANY_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

// Only one Tensix hardware semaphore is free on Blackhole, so every compute semaphore resolves to it; a
// second one in the same program would silently alias the first and is rejected up front.
TEST_F(ProgramSpecTestBlackhole, CPU_SecondComputeBoundSemaphoreIsRejected) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    BindSemaphoreToKernels(spec, "compute_sem_a", {"compute_kernel"});
    BindSemaphoreToKernels(spec, "compute_sem_b", {"compute_kernel"});

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("at most one compute semaphore")));
}

// The compute semaphore is seeded to 0 on the device by compute_kernel_hw_startup; the host cannot
// write the Tensix Sync Unit, so any other initial value is rejected up front.
TEST_F(ProgramSpecTestBlackhole, CPU_ComputeBoundSemaphoreWithNonzeroInitialValueIsRejected) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    BindSemaphoreToKernels(spec, "compute_sem", {"compute_kernel"});
    spec.semaphores.back().advanced_options.initial_value = 1;

    EXPECT_ANY_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

// The capacity (max_value) is the 4-bit hardware Max register: 1..15 on a compute semaphore, meaningless
// on a DM one.
TEST_F(ProgramSpecTestBlackhole, CPU_ComputeBoundSemaphoreMaxValueInRangeSucceeds) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    BindSemaphoreToKernels(spec, "compute_sem", {"compute_kernel"});
    spec.semaphores.back().advanced_options.max_value = 15;

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestBlackhole, CPU_ComputeBoundSemaphoreMaxValueAbove15IsRejected) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    BindSemaphoreToKernels(spec, "compute_sem", {"compute_kernel"});
    spec.semaphores.back().advanced_options.max_value = 16;

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("capacity is at most 15")));
}

TEST_F(ProgramSpecTestBlackhole, CPU_MaxValueOnDMBoundSemaphoreIsRejected) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    BindSemaphoreToKernels(spec, "dm_sem", {"dm_kernel"});
    spec.semaphores.back().advanced_options.max_value = 4;

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("not bound by a compute kernel")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
