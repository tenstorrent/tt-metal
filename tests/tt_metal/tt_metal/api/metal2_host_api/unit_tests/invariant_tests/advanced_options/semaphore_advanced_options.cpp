// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Local invariants of SemaphoreAdvancedOptions (advanced_options.hpp).

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <stdexcept>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "metal2_host_api/test_helpers.hpp"
#include "metal2_host_api/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::ProgramSpecTestGen1;
using test_helpers::ProgramSpecTestQuasar;

TEST_F(ProgramSpecTestQuasar, CPU_SemaphoreNonZeroInitialValueFailsOnQuasar) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    SemaphoreSpec sem;
    sem.unique_id = SemaphoreSpecName{"sem_0"};
    sem.target_nodes = NodeCoord{0, 0};
    sem.advanced_options = SemaphoreAdvancedOptions{.initial_value = 1};
    spec.semaphores = {sem};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("has initial_value=1 but only zero is supported on Quasar")));
}

TEST_F(ProgramSpecTestGen1, CPU_SemaphoresWithNonZeroInitialValueSucceedOnGen1) {
    // Gen1 accepts non-zero initial values (only Quasar rejects them).
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    SemaphoreSpec sem;
    sem.unique_id = SemaphoreSpecName{"sem_0"};
    sem.target_nodes = NodeCoord{0, 0};
    sem.advanced_options = SemaphoreAdvancedOptions{.initial_value = 3};
    spec.semaphores = {sem};

    spec.kernels[0].semaphore_bindings = {
        SemaphoreBinding{.semaphore_spec_name = SemaphoreSpecName{"sem_0"}, .accessor_name = "done_flag"}};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
