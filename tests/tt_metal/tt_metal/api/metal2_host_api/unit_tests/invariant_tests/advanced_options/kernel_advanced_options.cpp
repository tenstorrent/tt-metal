// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Local invariants of KernelAdvancedOptions (advanced_options.hpp): tensor binding sequences,
// per-node vararg counts and PrefetcherPipe bindings.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>

#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"
#include "metal2_host_api/test_helpers/prefetcher_pipe_test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindPipe;
using test_helpers::BindPipes;
using test_helpers::BindTensorParameterToKernel;
using test_helpers::KernelNamed;
using test_helpers::MakeFullPipeSpec;
using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalTensorParameter;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::MakeOtherPipeParameter;
using test_helpers::MakeSenderOnlySpec;
using test_helpers::other_param_name;
using test_helpers::pipe_param_name;
using test_helpers::PrefetcherPipeSpecTestQuasar;
using test_helpers::ProgramSpecTestGen1;
using test_helpers::ProgramSpecTestQuasar;

TEST_F(ProgramSpecTestGen1, CPU_TensorBindingSequenceUnknownMemberFails) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    spec.tensor_parameters = {MakeMinimalTensorParameter("t0")};
    BindTensorParameterToKernel(spec.kernels[0], "t0", "in0");
    spec.kernels[0].advanced_options.tensor_binding_sequences = {
        KernelAdvancedOptions::TensorBindingSequence{.sequence_name = "inputs", .members = {"in0", "missing"}},
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("references unknown tensor accessor_name 'missing'")));
}

TEST_F(ProgramSpecTestGen1, CPU_TensorBindingSequenceDuplicateMembersFails) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    spec.tensor_parameters = {MakeMinimalTensorParameter("t0")};
    BindTensorParameterToKernel(spec.kernels[0], "t0", "in0");
    spec.kernels[0].advanced_options.tensor_binding_sequences = {
        KernelAdvancedOptions::TensorBindingSequence{.sequence_name = "inputs", .members = {"in0", "in0"}},
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("has duplicate member 'in0'")));
}

TEST_F(ProgramSpecTestGen1, CPU_TensorBindingSequenceNameCollidesWithBindingFails) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    spec.tensor_parameters = {MakeMinimalTensorParameter("t0")};
    BindTensorParameterToKernel(spec.kernels[0], "t0", "in0");
    spec.kernels[0].advanced_options.tensor_binding_sequences = {
        KernelAdvancedOptions::TensorBindingSequence{.sequence_name = "in0", .members = {"in0"}},
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("collides with a TensorBinding accessor_name")));
}

TEST_F(ProgramSpecTestGen1, CPU_TensorBindingSequenceNameCollidesWithGeneratedTypeAliasFails) {
    // Codegen emits `using in0_t = TensorBindingToken<...>` for binding "in0". A sequence named
    // "in0_t" would emit `constexpr auto in0_t = ...` and fail to compile.
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    spec.tensor_parameters = {MakeMinimalTensorParameter("t0")};
    BindTensorParameterToKernel(spec.kernels[0], "t0", "in0");
    spec.kernels[0].advanced_options.tensor_binding_sequences = {
        KernelAdvancedOptions::TensorBindingSequence{.sequence_name = "in0_t", .members = {"in0"}},
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("collides with generated type alias 'in0_t'")));
}

TEST_F(ProgramSpecTestGen1, CPU_TensorBindingSequenceDuplicateSequenceNamesFails) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    spec.tensor_parameters = {MakeMinimalTensorParameter("t0")};
    BindTensorParameterToKernel(spec.kernels[0], "t0", "in0");
    spec.kernels[0].advanced_options.tensor_binding_sequences = {
        KernelAdvancedOptions::TensorBindingSequence{.sequence_name = "inputs", .members = {"in0"}},
        KernelAdvancedOptions::TensorBindingSequence{.sequence_name = "inputs", .members = {"in0"}},
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("has duplicate tensor binding sequence_name 'inputs'")));
}

TEST_F(ProgramSpecTestGen1, CPU_TensorBindingSequenceInvalidIdentifierFails) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    spec.tensor_parameters = {MakeMinimalTensorParameter("t0")};
    BindTensorParameterToKernel(spec.kernels[0], "t0", "in0");
    spec.kernels[0].advanced_options.tensor_binding_sequences = {
        KernelAdvancedOptions::TensorBindingSequence{.sequence_name = "has-dash", .members = {"in0"}},
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("tensor binding sequence_name 'has-dash' must be a valid C++ identifier")));
}

TEST_F(ProgramSpecTestQuasar, CPU_VarargPerNodeOverlapFails) {
    // Rule: overlapping entries in num_runtime_varargs_per_node are an error, even when
    // their counts agree. Overlap suggests a user mistake.
    NodeCoord node_a{0, 0};
    NodeCoord node_b{1, 0};
    NodeRangeSet both{std::vector<NodeRange>{NodeRange{node_a, node_a}, NodeRange{node_b, node_b}}};

    ProgramSpec spec;
    spec.name = "vararg_overlap_test";
    auto kernel = MakeMinimalGen2DMKernel("dm_kernel");
    kernel.advanced_options = KernelAdvancedOptions{
        .num_runtime_varargs_per_node = Table<Nodes, uint32_t>{{both, 3}, {node_a, 3}},  // node_a listed twice
    };
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit_0", both, {"dm_kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("overlapping entries")));
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_EmptyAccessorGroupFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    KernelNamed(spec, "sender").advanced_options.prefetcher_pipe_bindings[0].pipe_parameter_names.clear();
    EXPECT_SPEC_REJECTED(spec, "Kernel 'sender' PrefetcherPipe accessor 'weights' names no PrefetcherPipeParameter");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_SamePipeTwiceInOneAccessorFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    KernelNamed(spec, "sender").advanced_options.prefetcher_pipe_bindings[0].pipe_parameter_names = {
        pipe_param_name, pipe_param_name};
    EXPECT_SPEC_REJECTED(spec, "Kernel 'sender' binds PrefetcherPipeParameter 'weights' more than once");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_DuplicateAccessorNameFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    spec.advanced_options.prefetcher_pipe_parameters.push_back(MakeOtherPipeParameter());
    KernelNamed(spec, "sender")
        .advanced_options.prefetcher_pipe_bindings.push_back(BindPipes({other_param_name}, "weights"));
    EXPECT_SPEC_REJECTED(spec, "Kernel 'sender' has duplicate PrefetcherPipe accessor_name 'weights'");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_InvalidAccessorNameFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    KernelNamed(spec, "sender").advanced_options.prefetcher_pipe_bindings[0].accessor_name = "1weights";
    EXPECT_SPEC_REJECTED(spec, "PrefetcherPipe accessor_name '1weights' must be a valid C++ identifier");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_SamePipeBoundTwiceInOneKernelFails) {
    ProgramSpec spec = MakeSenderOnlySpec();
    KernelNamed(spec, "sender").advanced_options.prefetcher_pipe_bindings.push_back(BindPipe("weights_again"));
    EXPECT_SPEC_REJECTED(spec, "Kernel 'sender' binds PrefetcherPipeParameter 'weights' more than once");
}

TEST_F(PrefetcherPipeSpecTestQuasar, CPU_ComputeKernelBindingPipeFails) {
    ProgramSpec spec = MakeFullPipeSpec();
    KernelNamed(spec, "compute").advanced_options.prefetcher_pipe_bindings.push_back(BindPipe());
    EXPECT_SPEC_REJECTED(
        spec, "Kernel 'compute' binds PrefetcherPipeParameter(s) (accessor 'weights') but is a compute kernel");
}

}  // namespace
}  // namespace tt::tt_metal::experimental
