// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>
#include <vector>

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

namespace tt::tt_metal::experimental::metal2_host_api {

//------------------------------------------------
// MeshWorkloadSpec
//------------------------------------------------

// A MeshWorkloadSpec describes the immutable properties of a MeshWorkload:
// a collection of (MeshCoordinateRange, ProgramSpec) pairs.
//
// Each ProgramSpec is instantiated as a Program on the devices in its range.
// - Each device range must fit within the target mesh (need not cover the entire mesh)
// - Device ranges must not overlap (at most one ProgramSpec per device)
// - A valid MeshWorkloadSpec must contain at least one program.
//
struct MeshWorkloadSpec {
    // A ProgramSpec paired with its target placement on the mesh.
    struct ProgramPlacement {
        ProgramSpec program;
        distributed::MeshCoordinateRange target_range;
    };
    std::vector<ProgramPlacement> programs;
};

//------------------------------------------------
// Temporary Metal 2.0 API
// (will become a MeshWorkload constructor post-experimental)
//------------------------------------------------

// Create a MeshWorkload from a MeshWorkloadSpec.
//
// Each ProgramSpec is converted to a Program via MakeProgramFromSpec and added
// to the MeshWorkload for its corresponding MeshCoordinateRange.
//
// JIT compilation of kernels is deferred to first enqueue.
//
distributed::MeshWorkload MakeMeshWorkloadFromSpec(
    const distributed::MeshDevice& mesh_device, const MeshWorkloadSpec& spec, bool skip_validation = false);

}  // namespace tt::tt_metal::experimental::metal2_host_api
