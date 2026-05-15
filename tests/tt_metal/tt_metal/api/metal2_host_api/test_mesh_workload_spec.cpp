// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//---------------------------------------------------------------------------------
// Unit tests for the Metal 2.0 Host API: MeshWorkloadSpec and MakeMeshWorkloadFromSpec
// These tests use a mock Quasar device for API-level validation.
//---------------------------------------------------------------------------------

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <optional>

#include <tt-metalium/experimental/metal2_host_api/mesh_workload_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/mock_device.hpp>

#include "test_helpers.hpp"

namespace tt::tt_metal::experimental::metal2_host_api {
namespace {

using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::ScopedSlowDispatchOverride;

// Test fixture mirrors ProgramSpecTestQuasar: mock Quasar device under slow dispatch.
class MeshWorkloadSpecTestQuasar : public ::testing::Test {
protected:
    void SetUp() override {
        slow_dispatch_override_.emplace();
        experimental::configure_mock_mode(tt::ARCH::QUASAR, 1);
        mesh_device_ = distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::MeshShape{1, 1}));
    }
    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
            mesh_device_.reset();
        }
        experimental::disable_mock_mode();
        slow_dispatch_override_.reset();
    }

    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    std::optional<ScopedSlowDispatchOverride> slow_dispatch_override_;
};

// An empty MeshWorkloadSpec is a degenerate input — reject at the API boundary
// with a specific error rather than letting it propagate downstream.
TEST_F(MeshWorkloadSpecTestQuasar, EmptyMeshWorkloadSpecFails) {
    MeshWorkloadSpec spec;  // No programs.

    EXPECT_THAT(
        [&] { MakeMeshWorkloadFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("A MeshWorkloadSpec must contain at least one ProgramSpec")));
}

// Happy path: a single ProgramSpec covering the full mesh (SPMD convention) succeeds
// and produces a MeshWorkload containing exactly that program.
TEST_F(MeshWorkloadSpecTestQuasar, SingleProgramSPMDSucceeds) {
    MeshWorkloadSpec spec;
    spec.programs.push_back({
        .program = MakeMinimalValidProgramSpec(),
        .target_range = distributed::MeshCoordinateRange(mesh_device_->shape()),
    });

    distributed::MeshWorkload workload = MakeMeshWorkloadFromSpec(*mesh_device_, spec);

    EXPECT_EQ(workload.get_programs().size(), 1u);
}

// Ranges outside the target mesh produce a clear API-boundary error rather than
// propagating to a cryptic downstream failure at enqueue time.
TEST_F(MeshWorkloadSpecTestQuasar, OutOfBoundsRangeFails) {
    MeshWorkloadSpec spec;
    // (1, 1) is out of bounds for a {1, 1} mesh — valid coords are only (0, 0).
    spec.programs.push_back({
        .program = MakeMinimalValidProgramSpec(),
        .target_range = distributed::MeshCoordinateRange(distributed::MeshCoordinate(1, 1)),
    });

    EXPECT_THAT(
        [&] { MakeMeshWorkloadFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("out of bounds")));
}

// ProgramSpec validation failures must propagate through MakeMeshWorkloadFromSpec —
// proves the skip_validation=false default isn't silently dropped on the way in.
TEST_F(MeshWorkloadSpecTestQuasar, InvalidProgramSpecPropagatesValidationFailure) {
    ProgramSpec invalid_spec;  // Empty kernels — known to fail MakeProgramFromSpec validation.
    invalid_spec.program_id = "empty_program";

    MeshWorkloadSpec spec;
    spec.programs.push_back({
        .program = invalid_spec,
        .target_range = distributed::MeshCoordinateRange(mesh_device_->shape()),
    });

    EXPECT_THAT(
        [&] { MakeMeshWorkloadFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("A ProgramSpec must have at least one KernelSpec")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental::metal2_host_api
