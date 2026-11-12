// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// JIT-compile smoke tests for GlobalSemaphore bindings (mock Wormhole and mock Quasar): a data-movement
// kernel constructs GlobalSemaphore(global_sem::<accessor>) and calls every on-chip method.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>

#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestGen1;
using test_helpers::ProgramSpecTestQuasar;

constexpr const char* kKernelSource = R"(
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/global_semaphore.h"

void kernel_main() {
    GlobalSemaphore first(global_sem::first);
    GlobalSemaphore second(global_sem::second);
    Noc noc;
    first.up();
    first.up(noc, 1, 1, 2);
    first.wait_min(1);
    first.down(1);
    second.set(0);
    second.wait(0);
    second.set_multicast(noc, 1, 1, 2, 2, 4);
    second.inc_multicast(noc, 1, 1, 2, 2, 1, 4);
    volatile uint32_t value = first.value() + second.value();
    (void)value;
}
)";

ProgramSpec MakeSpec(KernelSpec dm_kernel) {
    dm_kernel.source = KernelSpec::SourceCode{kKernelSource};
    for (const char* name : {"first", "second"}) {
        dm_kernel.global_semaphore_bindings.push_back(GlobalSemaphoreBinding{
            .global_semaphore_parameter_name = GlobalSemaphoreParamName{name}, .accessor_name = name});
    }
    ProgramSpec spec;
    spec.name = "global_semaphore_smoke";
    spec.global_semaphore_parameters = {
        GlobalSemaphoreParameter{.unique_id = GlobalSemaphoreParamName{"first"}},
        GlobalSemaphoreParameter{.unique_id = GlobalSemaphoreParamName{"second"}},
    };
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", NodeCoord{0, 0}, {"dm_kernel"})};
    spec.kernels = {std::move(dm_kernel)};
    return spec;
}

TEST_F(ProgramSpecTestGen1, CPU_GlobalSemaphoreBindingJITSmoke) {
    Program program = MakeProgramFromSpec(*mesh_device_, MakeSpec(MakeMinimalGen1DMKernel("dm_kernel")));
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

TEST_F(ProgramSpecTestQuasar, CPU_GlobalSemaphoreBindingJITSmoke) {
    Program program = MakeProgramFromSpec(*mesh_device_, MakeSpec(MakeMinimalGen2DMKernel("dm_kernel")));
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
