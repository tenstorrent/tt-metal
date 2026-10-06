// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Minimal ProgramSpecs that satisfy every invariant and build a Program.

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindTensorParameterToKernel;
using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::MakeMinimalTensorParameter;
using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::ProgramSpecTestGen1;
using test_helpers::ProgramSpecTestQuasar;

TEST_F(ProgramSpecTestQuasar, CPU_MinimalValidProgramSpecSucceeds) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestGen1, CPU_MinimalValidProgramSpecSucceeds) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestGen1, CPU_MinimalValidProgramSpecWithTensorParameterSucceeds) {
    // Positive baseline: a program with one TensorParameter bound to one kernel constructs
    // successfully. Exercises CollectSpecData validation, the host-side resolution helper
    // (TensorSpec → CTA payload using mesh_device.allocator() + virtual_core_from_logical_core),
    // TensorBinding address slot assignment, kernel ctor plumbing, and ProgramImpl registration.
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    spec.tensor_parameters = {MakeMinimalTensorParameter("input_tensor")};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
