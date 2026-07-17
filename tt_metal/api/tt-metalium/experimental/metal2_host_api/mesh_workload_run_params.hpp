// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

namespace tt::tt_metal::experimental::metal2_host_api {

//------------------------------------------------
// MeshWorkloadRunParams
//------------------------------------------------

// MeshWorkloadRunParams describes the mutable properties of a MeshWorkload:
// ProgramRunParams for each Program in the MeshWorkload, identified by ProgramSpec name.
//
// Analogous to ProgramRunParams, which describes the mutable properties of a single Program.
//
// - Every Program in the target MeshWorkload must have a corresponding entry.
// - Every entry must name a Program in the target MeshWorkload (no extras).
// - Each ProgramSpec name must appear at most once.
struct MeshWorkloadRunParams {
    // ProgramRunParams paired with the name of the ProgramSpec they configure.
    struct NamedProgramRunParams {
        ProgramSpecName program_spec_name;
        ProgramRunParams run_params;
    };
    std::vector<NamedProgramRunParams> programs;
};

//------------------------------------------------
// Temporary Metal 2.0 API
// (will become a MeshWorkload member function post-experimental)
//------------------------------------------------

// Configure the mutable parameters of every Program in a MeshWorkload.
// Each NamedProgramRunParams entry is dispatched to the corresponding Program
// (looked up by ProgramSpec name) via SetProgramRunParameters.
//
// COMPLETENESS: A NamedProgramRunParams must be specified for every Program in the
// MeshWorkload, exactly once.
//
// PRE-CONDITION: All Programs in the MeshWorkload must have been constructed via
// MakeProgramFromSpec (so that they carry the program_spec_name needed for lookup).
void SetMeshWorkloadRunParameters(distributed::MeshWorkload& workload, const MeshWorkloadRunParams& params);

}  // namespace tt::tt_metal::experimental::metal2_host_api
