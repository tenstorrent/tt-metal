// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/metal2_host_api/mesh_workload_run_params.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include <tt-logger/tt-logger.hpp>

#include <unordered_map>
#include <unordered_set>

namespace tt::tt_metal::experimental::metal2_host_api {

void SetMeshWorkloadRunParameters(distributed::MeshWorkload& workload, const MeshWorkloadRunParams& params) {
    log_debug(tt::LogMetal, "Setting MeshWorkloadRunParameters ({} programs)", params.programs.size());

    TT_FATAL(!params.programs.empty(), "MeshWorkloadRunParams must contain at least one entry");

    // Build name -> Program* lookup from the cached workload.
    auto& workload_programs = workload.get_programs();
    std::unordered_map<ProgramSpecName, Program*> name_to_program;
    name_to_program.reserve(workload_programs.size());
    for (auto& [range, program] : workload_programs) {
        const auto& spec_name = program.get_program_spec_name();
        TT_FATAL(
            spec_name.has_value(),
            "MeshWorkload contains a Program with no program_spec_name (not constructed via "
            "MakeProgramFromSpec); cannot use SetMeshWorkloadRunParameters");
        auto [it, inserted] = name_to_program.try_emplace(*spec_name, &program);
        TT_FATAL(inserted, "MeshWorkload contains duplicate program_spec_name '{}'", *spec_name);
    }

    // Validate every entry against the workload before applying any of them
    // (so a failure leaves the workload unchanged).
    std::unordered_set<ProgramSpecName> seen_names;
    seen_names.reserve(params.programs.size());
    for (const auto& entry : params.programs) {
        TT_FATAL(
            seen_names.insert(entry.program_spec_name).second,
            "MeshWorkloadRunParams contains duplicate entry for program '{}'",
            entry.program_spec_name);
        TT_FATAL(
            name_to_program.contains(entry.program_spec_name),
            "MeshWorkloadRunParams names program '{}' which is not in the target MeshWorkload",
            entry.program_spec_name);
    }
    TT_FATAL(
        seen_names.size() == name_to_program.size(),
        "MeshWorkloadRunParams has {} entries but the MeshWorkload has {} programs; every program "
        "must have a corresponding entry exactly once",
        seen_names.size(),
        name_to_program.size());

    // Validation passed; apply the run params.
    for (const auto& entry : params.programs) {
        SetProgramRunParameters(*name_to_program.at(entry.program_spec_name), entry.run_params);
    }
}

}  // namespace tt::tt_metal::experimental::metal2_host_api
