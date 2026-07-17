// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//---------------------------------------------------------------------------------
// Unit tests for the Metal 2.0 Host API: MeshWorkloadRunParams and SetMeshWorkloadRunParameters
// These tests use a mock Quasar device for API-level validation.
//---------------------------------------------------------------------------------

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <optional>

#include <tt-metalium/experimental/metal2_host_api/mesh_workload_run_params.hpp>
#include <tt-metalium/experimental/metal2_host_api/mesh_workload_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>
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

class MeshWorkloadRunParamsTestQuasar : public ::testing::Test {
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

// Helper: minimal valid ProgramRunParams matching MakeMinimalValidProgramSpec's kernels
// (dm_kernel and compute_kernel, both with no RTAs).
inline ProgramRunParams MakeMinimalValidProgramRunParams() {
    NodeCoord node{0, 0};
    ProgramRunParams params;
    params.kernel_run_params.push_back(ProgramRunParams::KernelRunParams{
        .kernel_spec_name = "dm_kernel",
        .runtime_varargs = {{node, {}}},
        .common_runtime_varargs = {},
    });
    params.kernel_run_params.push_back(ProgramRunParams::KernelRunParams{
        .kernel_spec_name = "compute_kernel",
        .runtime_varargs = {{node, {}}},
        .common_runtime_varargs = {},
    });
    return params;
}

// Empty params is a degenerate input — reject at the API boundary.
TEST_F(MeshWorkloadRunParamsTestQuasar, EmptyParamsFails) {
    MeshWorkloadSpec spec;
    spec.programs.push_back({
        .program = MakeMinimalValidProgramSpec(),
        .target_range = distributed::MeshCoordinateRange(mesh_device_->shape()),
    });
    distributed::MeshWorkload workload = MakeMeshWorkloadFromSpec(*mesh_device_, spec);

    MeshWorkloadRunParams params;  // empty

    EXPECT_THAT(
        [&] { SetMeshWorkloadRunParameters(workload, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("MeshWorkloadRunParams must contain at least one entry")));
}

// Happy path: single program in workload, single matching entry in params.
TEST_F(MeshWorkloadRunParamsTestQuasar, MatchingNameSucceeds) {
    MeshWorkloadSpec spec;
    spec.programs.push_back({
        .program = MakeMinimalValidProgramSpec(),
        .target_range = distributed::MeshCoordinateRange(mesh_device_->shape()),
    });
    distributed::MeshWorkload workload = MakeMeshWorkloadFromSpec(*mesh_device_, spec);

    MeshWorkloadRunParams params;
    params.programs.push_back({
        .program_spec_name = "test_program",  // matches MakeMinimalValidProgramSpec's program_id
        .run_params = MakeMinimalValidProgramRunParams(),
    });

    EXPECT_NO_THROW(SetMeshWorkloadRunParameters(workload, params));
}

// An entry naming a program not in the workload is rejected at validation.
TEST_F(MeshWorkloadRunParamsTestQuasar, UnknownProgramNameFails) {
    MeshWorkloadSpec spec;
    spec.programs.push_back({
        .program = MakeMinimalValidProgramSpec(),
        .target_range = distributed::MeshCoordinateRange(mesh_device_->shape()),
    });
    distributed::MeshWorkload workload = MakeMeshWorkloadFromSpec(*mesh_device_, spec);

    MeshWorkloadRunParams params;
    params.programs.push_back({
        .program_spec_name = "nonexistent_program",
        .run_params = MakeMinimalValidProgramRunParams(),
    });

    EXPECT_THAT(
        [&] { SetMeshWorkloadRunParameters(workload, params); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("not in the target MeshWorkload")));
}

// Two entries for the same program name should be rejected as a duplicate.
TEST_F(MeshWorkloadRunParamsTestQuasar, DuplicateEntryFails) {
    MeshWorkloadSpec spec;
    spec.programs.push_back({
        .program = MakeMinimalValidProgramSpec(),
        .target_range = distributed::MeshCoordinateRange(mesh_device_->shape()),
    });
    distributed::MeshWorkload workload = MakeMeshWorkloadFromSpec(*mesh_device_, spec);

    MeshWorkloadRunParams params;
    params.programs.push_back({
        .program_spec_name = "test_program",
        .run_params = MakeMinimalValidProgramRunParams(),
    });
    params.programs.push_back({
        .program_spec_name = "test_program",  // duplicate
        .run_params = MakeMinimalValidProgramRunParams(),
    });

    EXPECT_THAT(
        [&] { SetMeshWorkloadRunParameters(workload, params); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("duplicate entry for program")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental::metal2_host_api
