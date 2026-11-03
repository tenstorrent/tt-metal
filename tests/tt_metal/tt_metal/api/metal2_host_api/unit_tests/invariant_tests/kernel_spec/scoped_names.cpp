// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Local invariants of scoped names (tech_reports/NamedKernelArgs/scoped_names.md): any kernel-visible name
// may be written "<scope>.<name>"; there is one level of scope; names in one scope are unique across
// all kinds; and a scope may not be named after a generated namespace.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <stdexcept>
#include <string>
#include <string_view>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "impl/metal2_host_api/scoped_name.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindTensorParameterToKernel;
using test_helpers::MakeMinimalTensorParameter;
using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::ProgramSpecTestQuasar;

void BindSemaphore(ProgramSpec& spec, const char* semaphore_name, std::string accessor_name) {
    spec.semaphores.push_back(
        SemaphoreSpec{.unique_id = SemaphoreSpecName{semaphore_name}, .target_nodes = NodeCoord{0, 0}});
    spec.kernels[0].semaphore_bindings.push_back(SemaphoreBinding{
        .semaphore_spec_name = SemaphoreSpecName{semaphore_name}, .accessor_name = std::move(accessor_name)});
}

// MakeMinimalValidProgramSpec's kernels[0] binds DFB "dfb_0" as "input_dfb".

TEST_F(ProgramSpecTestQuasar, CPU_ScopedNamesOfEveryKindSucceed) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    KernelSpec& kernel = spec.kernels[0];
    kernel.runtime_arg_schema.runtime_arg_names = {"op.num_tiles"};
    kernel.runtime_arg_schema.common_runtime_arg_names = {"op.batch"};
    kernel.compile_time_args = {{"op.Ht", 8}};
    kernel.dfb_bindings[0].accessor_name = "op.in";
    BindSemaphore(spec, "sem_0", "op.ready");
    spec.tensor_parameters = {MakeMinimalTensorParameter("t_0")};
    BindTensorParameterToKernel(kernel, "t_0", "op.t");
    kernel.advanced_options.tensor_binding_sequences = {
        KernelAdvancedOptions::TensorBindingSequence{.sequence_name = "op.inputs", .members = {"op.t"}},
    };

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_SameNameUnscopedAndScopedSucceeds) {
    // Unscoped names live in their kind namespaces (args::count, dfb::input_dfb), apart from any scope.
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].runtime_arg_schema.runtime_arg_names = {"count", "op.count", "op.input_dfb"};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_SameNameInTwoScopesSucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].runtime_arg_schema.runtime_arg_names = {"a.count", "b.count"};
    spec.kernels[0].compile_time_args = {{"a.Ht", 8}, {"b.Ht", 16}};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_ScopedNameCollisionAcrossKindsFails) {
    // A scope holds every kind in one struct, so a scoped RTA and a scoped DFB can't share a name.
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].runtime_arg_schema.runtime_arg_names = {"op.x"};
    spec.kernels[0].dfb_bindings[0].accessor_name = "op.x";

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("has two members named 'x' in scope 'op'")));
}

TEST_F(ProgramSpecTestQuasar, CPU_ScopedNameCollidesWithTensorTypeAliasFails) {
    // A scoped tensor binding 'op.t' also emits the type alias op::t_t.
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.tensor_parameters = {MakeMinimalTensorParameter("t_0")};
    BindTensorParameterToKernel(spec.kernels[0], "t_0", "op.t");
    spec.kernels[0].runtime_arg_schema.runtime_arg_names = {"op.t_t"};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("has two members named 't_t' in scope 'op'")));
}

TEST_F(ProgramSpecTestQuasar, CPU_ScopedNameCollidesWithGeneratedLookupFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].runtime_arg_schema.runtime_arg_names = {"op.get_dfb_token_if_present"};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("has two members named 'get_dfb_token_if_present' in scope 'op'")));
}

TEST_F(ProgramSpecTestQuasar, CPU_ReservedScopeNameFails) {
    // Every reserved name, so the list is pinned
    for (std::string_view scope : RESERVED_SCOPE_NAMES) {
        const std::string name = std::string(scope) + ".x";
        SCOPED_TRACE(name);
        ProgramSpec spec = MakeMinimalValidProgramSpec();
        spec.kernels[0].runtime_arg_schema.runtime_arg_names = {name};

        EXPECT_THAT(
            [&] { MakeProgramFromSpec(*mesh_device_, spec); },
            ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("uses the reserved scope name")));
    }
}

TEST_F(ProgramSpecTestQuasar, CPU_ScopedNameMatchingItsScopeFails) {
    // A C++ class can't have a member with its own name.
    {
        ProgramSpec spec = MakeMinimalValidProgramSpec();
        spec.kernels[0].runtime_arg_schema.runtime_arg_names = {"op.op"};

        EXPECT_THAT(
            [&] { MakeProgramFromSpec(*mesh_device_, spec); },
            ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("two members named 'op' in scope 'op'")));
    }
    {
        // The tensor binding's generated type alias, t_t::t_t, would take the scope's name.
        ProgramSpec spec = MakeMinimalValidProgramSpec();
        spec.tensor_parameters = {MakeMinimalTensorParameter("t_0")};
        BindTensorParameterToKernel(spec.kernels[0], "t_0", "t_t.t");

        EXPECT_THAT(
            [&] { MakeProgramFromSpec(*mesh_device_, spec); },
            ::testing::ThrowsMessage<std::runtime_error>(
                ::testing::HasSubstr("two members named 't_t' in scope 't_t'")));
    }
    {
        // So would the scope's generated lookup of the same name.
        ProgramSpec spec = MakeMinimalValidProgramSpec();
        spec.kernels[0].runtime_arg_schema.runtime_arg_names = {"get_dfb_token_if_present.x"};

        EXPECT_THAT(
            [&] { MakeProgramFromSpec(*mesh_device_, spec); },
            ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
                "two members named 'get_dfb_token_if_present' in scope 'get_dfb_token_if_present'")));
    }
}

TEST_F(ProgramSpecTestQuasar, CPU_MalformedScopedArgNameFails) {
    // One level of scope, with a C++ identifier on each side of the separator
    for (const char* name : {"a.b.c", ".x", "x.", "op.int", "1op.x"}) {
        SCOPED_TRACE(name);
        ProgramSpec spec = MakeMinimalValidProgramSpec();
        spec.kernels[0].runtime_arg_schema.runtime_arg_names = {name};

        EXPECT_THAT(
            [&] { MakeProgramFromSpec(*mesh_device_, spec); },
            ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("must be a valid C++ identifier")));
    }
}

TEST_F(ProgramSpecTestQuasar, CPU_MalformedScopedAccessorNameFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].dfb_bindings[0].accessor_name = "a.b.c";

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("must be a valid C++ identifier")));
}

TEST_F(ProgramSpecTestQuasar, CPU_SequenceMemberFromAnotherScopeFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.tensor_parameters = {MakeMinimalTensorParameter("t_0")};
    BindTensorParameterToKernel(spec.kernels[0], "t_0", "a.t");
    spec.kernels[0].advanced_options.tensor_binding_sequences = {
        KernelAdvancedOptions::TensorBindingSequence{.sequence_name = "b.inputs", .members = {"a.t"}},
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("has member 'a.t' from a different scope")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
