// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

using MeshQuiesceTestSuite = GenericMeshDeviceFixture;

// The test succeeds if it completes without hanging.
TEST_F(MeshQuiesceTestSuite, QuiesceSubmeshesAllowsAlternatingWorkloads) {
    if (mesh_device_->num_devices() < 2) {
        GTEST_SKIP() << "Requires at least 2 devices";
    }

    // Create two evenly sized submeshes (split along columns if possible, else rows).
    MeshShape parent_shape = mesh_device_->shape();
    std::optional<MeshShape> sub_shape;
    if (parent_shape.dims() == 2 && (parent_shape[1] % 2 == 0)) {
        sub_shape = MeshShape(parent_shape[0], parent_shape[1] / 2);
    } else if (parent_shape.dims() == 2 && (parent_shape[0] % 2 == 0)) {
        sub_shape = MeshShape(parent_shape[0] / 2, parent_shape[1]);
    }

    if (!sub_shape.has_value()) {
        GTEST_SKIP() << "Mesh shape is not evenly splittable into two submeshes";
    }

    auto submeshes = mesh_device_->create_submeshes(*sub_shape);
    ASSERT_EQ(submeshes.size(), 2u);
    auto submesh = submeshes.front();

    // Single-core no-op program for submesh
    Program submesh_program = CreateProgram();
    CoreCoord single_core = {0, 0};
    CreateKernel(submesh_program, "tt_metal/kernels/compute/blank.cpp", single_core, ComputeConfig{});
    MeshWorkload submesh_workload;
    submesh_workload.add_program(MeshCoordinateRange(submesh->shape()), std::move(submesh_program));

    // Single-core no-op program for parent mesh
    Program parent_program = CreateProgram();
    CreateKernel(parent_program, "tt_metal/kernels/compute/blank.cpp", single_core, ComputeConfig{});
    MeshWorkload parent_workload;
    parent_workload.add_program(MeshCoordinateRange(mesh_device_->shape()), std::move(parent_program));

    // 1) Run on submesh (non-blocking)
    EnqueueMeshWorkload(submesh->mesh_command_queue(), submesh_workload, /*blocking=*/false);

    // 2) Quiesce all submeshes from the parent
    mesh_device_->quiesce_submeshes();

    // 3) Run on parent (non-blocking)
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), parent_workload, /*blocking=*/false);

    // 4) Quiesce again
    mesh_device_->quiesce_submeshes();

    // 5) Run again on the same submesh (non-blocking) and finish to ensure completion
    EnqueueMeshWorkload(submesh->mesh_command_queue(), submesh_workload, /*blocking=*/false);
    Finish(submesh->mesh_command_queue());
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
