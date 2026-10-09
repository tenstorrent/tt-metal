// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ProgramSpec structural invariant (program_spec.hpp): every name resolves. Each binding, WorkUnitSpec::kernels
// entry and DataflowBufferSpec::borrowed_from names a declared spec.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <stdexcept>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindTensorParameterToKernel;
using test_helpers::MakeBorrowedDFBProgramSpec;
using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestGen1;
using test_helpers::ProgramSpecTestQuasar;

TEST_F(ProgramSpecTestQuasar, CPU_KernelReferencesUnknownDFBFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel = MakeMinimalGen2DMKernel("kernel");
    // Bind to a DFB that doesn't exist
    kernel.dfb_bindings.push_back(ProducerOf(DFBSpecName{"nonexistent_dfb"}, "accessor"));

    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'kernel' references unknown DFB 'nonexistent_dfb'")));
}

TEST_F(ProgramSpecTestQuasar, CPU_KernelSemaphoreBindingUnknownSemaphoreFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    SemaphoreBinding binding;
    binding.semaphore_spec_name = SemaphoreSpecName{"missing_sem"};
    binding.accessor_name = "my_sem";
    spec.kernels[0].semaphore_bindings = {binding};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("references unknown semaphore 'missing_sem'")));
}

TEST_F(ProgramSpecTestQuasar, CPU_UnknownScratchpadReferenceFails) {
    // A scratchpad_binding referencing a scratchpad_spec_name that isn't declared in spec.scratchpads.
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // No spec.scratchpads declared, but the kernel binds one.
    spec.kernels[0].scratchpad_bindings = {KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"missing_scratch"}, .accessor_name = "s"}};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("references unknown scratchpad")));
}

TEST_F(ProgramSpecTestGen1, CPU_KernelReferencesUnknownTensorParameterFails) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    // Reference a TensorParameter that doesn't exist in the program.
    BindTensorParameterToKernel(spec.kernels[0], "nonexistent_tensor", "input_ta");

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("references unknown TensorParameter 'nonexistent_tensor'")));
}

TEST_F(ProgramSpecTestQuasar, CPU_WorkUnitSpecReferencesUnknownKernelFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel = MakeMinimalGen2DMKernel("real_kernel");
    spec.kernels = {kernel};

    // WorkUnit references a kernel that doesn't exist
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"nonexistent_kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("WorkUnitSpec 'work_unit' references unknown kernel 'nonexistent_kernel'")));
}

TEST_F(ProgramSpecTestQuasar, CPU_BorrowedMemoryDFBUnknownTensorParameterFails) {
    ProgramSpec spec = MakeBorrowedDFBProgramSpec("borrowed_tensor");
    // Re-target the DFB at a TensorParameter that wasn't declared.
    spec.dataflow_buffers[0].borrowed_from = TensorParamName{"nonexistent_tensor"};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("borrows memory from TensorParameter 'nonexistent_tensor'")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
