// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/metal2_host_api/mesh_workload_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include <tt-logger/tt-logger.hpp>

namespace tt::tt_metal::experimental::metal2_host_api {

distributed::MeshWorkload MakeMeshWorkloadFromSpec(
    const distributed::MeshDevice& mesh_device, const MeshWorkloadSpec& spec, bool skip_validation) {
    log_debug(tt::LogMetal, "Creating MeshWorkload from MeshWorkloadSpec ({} programs)", spec.programs.size());

    TT_FATAL(!spec.programs.empty(), "A MeshWorkloadSpec must contain at least one ProgramSpec");

    // Pre-check that each range fits within the target mesh.
    // (Overlap is caught downstream by MeshWorkload::add_program with a clear message;
    // out-of-bounds is caught here because downstream enqueue surfaces it as a cryptic
    // "MeshDeviceViewImpl::is_local rejects unknown coordinate" failure.)
    const auto mesh_range = distributed::MeshCoordinateRange(mesh_device.shape());
    for (const auto& placement : spec.programs) {
        TT_FATAL(
            mesh_range.contains(placement.target_range),
            "MeshWorkloadSpec range {} is out of bounds for mesh of shape {}",
            placement.target_range,
            mesh_device.shape());
    }

    distributed::MeshWorkload mesh_workload;
    for (const auto& placement : spec.programs) {
        mesh_workload.add_program(
            placement.target_range, MakeProgramFromSpec(mesh_device, placement.program, skip_validation));
    }
    return mesh_workload;
}

}  // namespace tt::tt_metal::experimental::metal2_host_api
