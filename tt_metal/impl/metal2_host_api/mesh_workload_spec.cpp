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

    distributed::MeshWorkload mesh_workload;
    for (const auto& [range, program_spec] : spec.programs) {
        mesh_workload.add_program(range, MakeProgramFromSpec(mesh_device, program_spec, skip_validation));
    }
    return mesh_workload;
}

}  // namespace tt::tt_metal::experimental::metal2_host_api
