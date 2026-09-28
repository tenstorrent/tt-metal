// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Local invariants of KernelSpec::TensorBinding and KernelSpec::tensor_bindings (kernel_spec.hpp), plus
// the rule that each binding category has its own accessor-name namespace.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <stdexcept>
#include <string>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "metal2_host_api/test_helpers.hpp"
#include "metal2_host_api/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindTensorParameterToKernel;
using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::MakeMinimalTensorParameter;
using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::ProgramSpecTestGen1;
using test_helpers::ProgramSpecTestQuasar;

TEST_F(ProgramSpecTestGen1, CPU_DuplicateTensorAccessorNameWithinKernelFails) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    spec.tensor_parameters = {
        MakeMinimalTensorParameter("input_tensor"),
        MakeMinimalTensorParameter("output_tensor"),
    };
    // Two bindings on the same kernel under the same accessor_name — illegal.
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "same_accessor");
    BindTensorParameterToKernel(spec.kernels[0], "output_tensor", "same_accessor");

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("has duplicate tensor accessor_name 'same_accessor'")));
}

TEST_F(ProgramSpecTestGen1, CPU_InvalidTensorAccessorNameFails) {
    // Smoke-tests the IsValidCppIdentifier / length checks on tensor accessor names. The checks
    // are the same ones DFB / Semaphore use; one bad name of each kind here is sufficient
    // (full identifier coverage lives in the DFB version of this test).
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    spec.tensor_parameters = {MakeMinimalTensorParameter("input_tensor")};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "has-dash");  // not a valid identifier

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("tensor accessor_name 'has-dash' must be a valid C++ identifier")));

    spec = MakeMinimalGen1ValidProgramSpec();
    spec.tensor_parameters = {MakeMinimalTensorParameter("input_tensor")};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", std::string(MAX_ACCESSOR_NAME_LENGTH + 1, 'a'));

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("must be at most")));
}

TEST_F(ProgramSpecTestQuasar, CPU_TensorBindingOnComputeKernelIsAccepted) {
    // A tensor binding on a compute kernel is legal: the kernel constructs a LocalTensorAccessor
    // (NOC-free) from the binding token rather than a TensorAccessor. ValidateProgramSpec accepts it;
    // there is no host-side residency check. (The compile/dispatch path is proven in the HW test
    // LocalTensorAccessorBindingCompileComputeKernel.)
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.tensor_parameters = {MakeMinimalTensorParameter("t")};
    BindTensorParameterToKernel(spec.kernels[1], "t", "t_acc");  // kernels[1] == compute_kernel

    EXPECT_NO_THROW({ MakeProgramFromSpec(*mesh_device_, spec); });
}

TEST_F(ProgramSpecTestGen1, CPU_AccessorNamesAcrossCategoriesAreSeparateNamespaces) {
    // DFB / Semaphore / TensorAccessor accessor names live in separate namespaces (each gets
    // its own emitted namespace in kernel_bindings_generated.h: dfb::, sem::, tensor::). Reusing
    // the same identifier across categories within one kernel must be allowed.
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    spec.tensor_parameters = {MakeMinimalTensorParameter("input_tensor")};
    // The minimal program already has a DFB binding under accessor_name "input_dfb". Add a
    // semaphore and a tensor accessor, both also named "input_dfb" — the same string at a
    // C++ level — which should pass because they're in different namespaces.
    SemaphoreSpec sem;
    sem.unique_id = SemaphoreSpecName{"sem_0"};
    sem.target_nodes = NodeCoord{0, 0};
    spec.semaphores = {sem};
    spec.kernels[0].semaphore_bindings = {
        SemaphoreBinding{.semaphore_spec_name = SemaphoreSpecName{"sem_0"}, .accessor_name = "input_dfb"}};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_dfb");

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
