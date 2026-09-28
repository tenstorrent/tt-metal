// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// JIT-compile smoke tests for tensor bindings and tensor binding sequences (mock Wormhole).

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <type_traits>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers.hpp"
#include "metal2_host_api/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindTensorParameterToKernel;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::MakeMinimalTensorParameter;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestGen1;

// ============================================================================
// TensorParameter JIT Smoke Tests (Gen1 / WH)
// ============================================================================
// Codegen-path smoke test for the Metal 2.0 TensorAccessor binding feature. Ends in
// program.impl().compile, so the auto-generated kernel_bindings_generated.h (with its `tensor::` namespace)
// must be syntactically valid and compose correctly with the rest of the kernel build. Doesn't
// validate runtime behavior — catches regressions in codegen string-formatting, token type alias
// generation, and include-path resolution.
//
// DM-only by design: tensor_accessor.h pulls in NoC-using headers (dataflow_api_addrgen.h,
// pages_address_iterator.h with ASSERT) that don't compile on TRISC. There are no compute-kernel
// uses of TensorAccessor in the wild; the device-side library was built for DM kernels.
// Compute-kernel TensorAccessor bindings are unsupported in this PR; making them work would
// require restructuring tensor_accessor.h to isolate the constexpr-only parts.

TEST_F(ProgramSpecTestGen1, CPU_TensorAccessorBindingJITSmokeDMKernel) {
    // DM kernel constructs a TensorAccessor from a binding token + invokes a NoC-using method.
    // Exercises: tensor:: namespace token, type alias <name>_t, the token ctor and its deduction
    // guide, get_common_arg_val for the implicit base address.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "ta_smoke_dm";

    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.source = KernelSpec::SourceCode{R"(
void kernel_main() {
    TensorAccessor accessor(tensor::input_tensor);
    auto noc_addr = accessor.get_noc_addr(0);
    (void)noc_addr;
}

static_assert(tensor::get_token_if_present<"input_tensor">() == &tensor::input_tensor);
static_assert(tensor::get_token_if_present<"not_a_tensor">() == nullptr);
)"};

    spec.kernels = {dm_kernel};
    spec.tensor_parameters = {MakeMinimalTensorParameter("input_tensor")};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_tensor");
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

TEST_F(ProgramSpecTestGen1, CPU_TensorBindingSequenceSeveralMembersJITSmoke) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "tensor_binding_sequence_several";

    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.source = KernelSpec::SourceCode{R"(
#include <type_traits>
void kernel_main() {
    // Test-only: do not use std::tuple_element_t / remove_cv_t in real kernels unless necessary.
    using inputs_t = std::remove_cv_t<decltype(tensor::inputs)>;
    static_assert(std::tuple_size_v<inputs_t> == 3);
    static_assert(std::is_same_v<std::tuple_element_t<0, inputs_t>, tensor::in0_t>);
    static_assert(std::is_same_v<std::tuple_element_t<1, inputs_t>, tensor::in1_t>);
    static_assert(std::is_same_v<std::tuple_element_t<2, inputs_t>, tensor::in2_t>);
}
)"};
    dm_kernel.advanced_options.tensor_binding_sequences = {
        KernelAdvancedOptions::TensorBindingSequence{.sequence_name = "inputs", .members = {"in0", "in1", "in2"}},
    };

    spec.kernels = {dm_kernel};
    spec.tensor_parameters = {
        MakeMinimalTensorParameter("t0"),
        MakeMinimalTensorParameter("t1"),
        MakeMinimalTensorParameter("t2"),
    };
    BindTensorParameterToKernel(spec.kernels[0], "t0", "in0");
    BindTensorParameterToKernel(spec.kernels[0], "t1", "in1");
    BindTensorParameterToKernel(spec.kernels[0], "t2", "in2");
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

TEST_F(ProgramSpecTestGen1, CPU_TensorBindingSequenceEmptyMembersJITSmoke) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "tensor_binding_sequence_empty";

    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.source = KernelSpec::SourceCode{R"(
#include <type_traits>
void kernel_main() {
    // Test-only: do not use std::tuple_element_t / remove_cv_t in real kernels unless necessary.
    static_assert(std::tuple_size_v<decltype(tensor::empty)> == 0);
    static_assert(std::is_same_v<std::remove_cv_t<decltype(tensor::empty)>, std::tuple<>>);
}
)"};
    dm_kernel.advanced_options.tensor_binding_sequences = {
        KernelAdvancedOptions::TensorBindingSequence{.sequence_name = "empty", .members = {}},
    };

    spec.kernels = {dm_kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

TEST_F(ProgramSpecTestGen1, CPU_TensorBindingSequenceSingletonMembersJITSmoke) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "tensor_binding_sequence_singleton";

    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.source = KernelSpec::SourceCode{R"(
#include <type_traits>
void kernel_main() {
    // Test-only: do not use std::tuple_element_t / remove_cv_t in real kernels unless necessary.
    using solo_t = std::remove_cv_t<decltype(tensor::solo)>;
    static_assert(std::tuple_size_v<solo_t> == 1);
    static_assert(std::is_same_v<std::tuple_element_t<0, solo_t>, tensor::in0_t>);
}
)"};
    dm_kernel.advanced_options.tensor_binding_sequences = {
        KernelAdvancedOptions::TensorBindingSequence{.sequence_name = "solo", .members = {"in0"}},
    };

    spec.kernels = {dm_kernel};
    spec.tensor_parameters = {MakeMinimalTensorParameter("t0")};
    BindTensorParameterToKernel(spec.kernels[0], "t0", "in0");
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

TEST_F(ProgramSpecTestGen1, CPU_TensorBindingSequenceSameBindingInTwoSequencesJITSmoke) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "tensor_binding_sequence_shared_member";

    auto dm_kernel = MakeMinimalGen1DMKernel("dm_kernel");
    dm_kernel.source = KernelSpec::SourceCode{R"(
#include <type_traits>
void kernel_main() {
    // Test-only: do not use std::tuple_element_t / remove_cv_t in real kernels unless necessary.
    using g0_t = std::remove_cv_t<decltype(tensor::g0)>;
    using g1_t = std::remove_cv_t<decltype(tensor::g1)>;
    static_assert(std::tuple_size_v<g0_t> == 1);
    static_assert(std::tuple_size_v<g1_t> == 1);
    static_assert(std::is_same_v<std::tuple_element_t<0, g0_t>, tensor::in0_t>);
    static_assert(std::is_same_v<std::tuple_element_t<0, g1_t>, tensor::in0_t>);
}
)"};
    dm_kernel.advanced_options.tensor_binding_sequences = {
        KernelAdvancedOptions::TensorBindingSequence{.sequence_name = "g0", .members = {"in0"}},
        KernelAdvancedOptions::TensorBindingSequence{.sequence_name = "g1", .members = {"in0"}},
    };

    spec.kernels = {dm_kernel};
    spec.tensor_parameters = {MakeMinimalTensorParameter("t0")};
    BindTensorParameterToKernel(spec.kernels[0], "t0", "in0");
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

TEST_F(ProgramSpecTestGen1, CPU_TensorBindingSequenceOnComputeKernelJITSmoke) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    ASSERT_TRUE(spec.kernels[1].is_compute_kernel());

    spec.tensor_parameters = {MakeMinimalTensorParameter("t0"), MakeMinimalTensorParameter("t1")};
    BindTensorParameterToKernel(spec.kernels[1], "t0", "in0");
    BindTensorParameterToKernel(spec.kernels[1], "t1", "in1");
    spec.kernels[1].advanced_options.tensor_binding_sequences = {
        KernelAdvancedOptions::TensorBindingSequence{.sequence_name = "inputs", .members = {"in0", "in1"}},
    };
    spec.kernels[1].source = KernelSpec::SourceCode{R"(
#include <type_traits>
void kernel_main() {
    // Test-only: do not use std::tuple_element_t / remove_cv_t in real kernels unless necessary.
    using inputs_t = std::remove_cv_t<decltype(tensor::inputs)>;
    static_assert(std::tuple_size_v<inputs_t> == 2);
    static_assert(std::is_same_v<std::tuple_element_t<0, inputs_t>, tensor::in0_t>);
    static_assert(std::is_same_v<std::tuple_element_t<1, inputs_t>, tensor::in1_t>);
}
)"};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

TEST_F(ProgramSpecTestGen1, CPU_TensorBindingSequenceNameEqualsDfbAccessorJITSmoke) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    spec.tensor_parameters = {MakeMinimalTensorParameter("t0")};
    BindTensorParameterToKernel(spec.kernels[0], "t0", "in0");
    // Minimal program already binds dfb accessor "input_dfb" on kernels[0].
    spec.kernels[0].advanced_options.tensor_binding_sequences = {
        KernelAdvancedOptions::TensorBindingSequence{.sequence_name = "input_dfb", .members = {"in0"}},
    };
    spec.kernels[0].source = KernelSpec::SourceCode{R"(
#include <type_traits>
void kernel_main() {
    // Test-only: do not use std::tuple_element_t / remove_cv_t in real kernels unless necessary.
    using input_dfb_t = std::remove_cv_t<decltype(tensor::input_dfb)>;
    static_assert(std::tuple_size_v<input_dfb_t> == 1);
    static_assert(std::is_same_v<std::tuple_element_t<0, input_dfb_t>, tensor::in0_t>);
    (void)dfb::input_dfb;
}
)"};

    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    EXPECT_NO_THROW(program.impl().compile(mesh_device_.get()));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
