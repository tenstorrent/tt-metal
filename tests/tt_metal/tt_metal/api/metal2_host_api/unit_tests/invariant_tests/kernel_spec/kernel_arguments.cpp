// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Local invariants of KernelSpec::compile_time_args and KernelSpec::RuntimeArgSchema (kernel_spec.hpp):
// names are C++ identifiers and do not repeat across CTAs, RTAs and CRTAs of one kernel.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <stdexcept>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::ProgramSpecTestQuasar;

TEST_F(ProgramSpecTestQuasar, CPU_NamedRuntimeArgsSucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].runtime_arg_schema.runtime_arg_names = {"input_ptr", "output_ptr"};
    spec.kernels[0].runtime_arg_schema.common_runtime_arg_names = {"tile_count"};
    spec.kernels[0].compile_time_args = {{"block_size", 64}};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_InvalidNamedRtaIdentifierFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].runtime_arg_schema.runtime_arg_names = {"int"};  // C++ keyword

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("named RTA name 'int' is not a valid C++ identifier")));
}

TEST_F(ProgramSpecTestQuasar, CPU_InvalidNamedCrtaIdentifierFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].runtime_arg_schema.common_runtime_arg_names = {"has-dash"};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("named CRTA name 'has-dash' is not a valid C++ identifier")));
}

TEST_F(ProgramSpecTestQuasar, CPU_NamedRtaCrtaCollisionFails) {
    // A single name cannot be both a named RTA and a named CRTA (they share the user namespace).
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].runtime_arg_schema.runtime_arg_names = {"count"};
    spec.kernels[0].runtime_arg_schema.common_runtime_arg_names = {"count"};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("naming collision: 'count' is declared as both a named RTA and a named CRTA")));
}

TEST_F(ProgramSpecTestQuasar, CPU_NamedRtaCtaCollisionFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].runtime_arg_schema.runtime_arg_names = {"block_size"};
    spec.kernels[0].compile_time_args = {{"block_size", 64}};  // same name as CTA

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("naming collision: 'block_size' is declared as both a named RTA and a named CTA")));
}

TEST_F(ProgramSpecTestQuasar, CPU_DifferentKernelsMayReuseArgNames) {
    // Collision rule is per-kernel. Two different kernels may have identically-named args.
    ProgramSpec spec = MakeMinimalValidProgramSpec();
    spec.kernels[0].runtime_arg_schema.runtime_arg_names = {"shared_name"};
    spec.kernels[1].runtime_arg_schema.runtime_arg_names = {"shared_name"};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_CompileTimeArgBindingsSucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Add compile-time arg bindings
    spec.kernels[0].compile_time_args = {{"arg1", 100}, {"arg2", 200}};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_RuntimeArgsSchemaSucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Add runtime args schema
    spec.kernels[0].advanced_options = KernelAdvancedOptions{.num_runtime_varargs = 3, .num_common_runtime_varargs = 2};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
