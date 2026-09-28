// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ProgramSpec structural invariant (program_spec.hpp): every declaration is used. Semaphores are exempt.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <stdexcept>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "metal2_host_api/test_helpers.hpp"
#include "metal2_host_api/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalTensorParameter;
using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestGen1;
using test_helpers::ProgramSpecTestQuasar;

TEST_F(ProgramSpecTestQuasar, CPU_KernelNotInAnyWorkUnitSpecFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel1 = MakeMinimalGen2DMKernel("kernel1");
    auto kernel2 = MakeMinimalGen2DMKernel("kernel2");  // Not in any work_unit!
    spec.kernels = {kernel1, kernel2};

    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel1"})};  // Only kernel1

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'kernel2' is not referenced by any WorkUnitSpec")));
}

TEST_F(ProgramSpecTestQuasar, CPU_DFBWithNoBindingsFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    // Create a kernel with no DFB bindings
    auto kernel = MakeMinimalGen2DMKernel("kernel");
    spec.kernels = {kernel};

    // Create a DFB that is never bound
    auto orphan_dfb = MakeMinimalDFB("orphan_dfb");
    spec.dataflow_buffers = {orphan_dfb};

    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("DFB 'orphan_dfb' is defined but not bound by any kernel")));
}

TEST_F(ProgramSpecTestQuasar, CPU_UnboundScratchpadFails) {
    // A ScratchpadSpec declared in spec.scratchpads that no kernel binds. An unbound scratchpad
    // reserves L1 no kernel can reach.
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"orphan_scratch"}, .size_per_node = 1024}};
    // No kernel binds it.

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("declared but not bound")));
}

TEST_F(ProgramSpecTestGen1, CPU_UnboundTensorParameterFails) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    // Declare a TensorParameter but don't bind it to any kernel.
    spec.tensor_parameters = {MakeMinimalTensorParameter("orphan_tensor")};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("TensorParameter 'orphan_tensor' is defined but not bound by any kernel")));
}

TEST_F(ProgramSpecTestQuasar, CPU_SemaphoresSucceed) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    SemaphoreSpec sem;
    sem.unique_id = SemaphoreSpecName{"sem_0"};
    sem.target_nodes = NodeCoord{0, 0};
    spec.semaphores = {sem};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
