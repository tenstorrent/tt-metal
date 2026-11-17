// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// JIT-compile smoke test for GlobalSemaphore bindings (mock Wormhole).

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestGen1;

TEST_F(ProgramSpecTestGen1, CPU_GlobalSemaphoreBindingJITSmoke) {
    // A DM kernel constructs a GlobalSemaphore from its binding token and uses it.
    KernelSpec writer = MakeMinimalGen1DMKernel("writer");
    writer.source = KernelSpec::SourceCode{R"(
#include "api/dataflow/global_semaphore.h"
void kernel_main() {
    experimental::GlobalSemaphore signal(global_sem::signal);
    signal.wait(1);
    signal.set(0);
    volatile uint32_t address = signal.address();
    (void)address;
}
)"};
    writer.advanced_options.global_semaphore_bindings = {
        {.global_semaphore_parameter_name = GlobalSemaphoreParamName{"sem"}, .accessor_name = "signal"}};

    ProgramSpec spec{.kernels = {writer}, .work_units = {MakeMinimalWorkUnit("work", NodeCoord{0, 0}, {"writer"})}};
    spec.advanced_options.global_semaphore_parameters = {{.unique_id = GlobalSemaphoreParamName{"sem"}}};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
