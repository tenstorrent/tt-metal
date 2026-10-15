// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <unordered_map>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt_stl/fmt.hpp>

#include "distributed/mesh_device_impl.hpp"
#include "distributed/mesh_workload_impl.hpp"
#include "impl/context/context_types.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/metal2_host_api/program_spec/collection/collect_metadata.hpp"
#include "impl/metal2_host_api/program_spec/construction/construct_program.hpp"
#include "impl/metal2_host_api/program_spec/validation/validate_spec.hpp"
#include "impl/program/program_impl.hpp"

namespace tt::tt_metal::experimental {

Program BuildProgramFromSpec(distributed::MeshDevice& mesh_device, const ProgramSpec& spec, bool skip_validation) {
    log_debug(tt::LogMetal, "Creating Program from ProgramSpec ({})", spec.name);
    MetalContext& metal_ctx = mesh_device.impl().metal_context();

    // Step 1a: Collect derived data (builds lookup tables, checks name uniqueness and references)
    CollectedSpecData collected = CollectSpecData(spec);

    // Step 1b: Validate semantic rules (can be skipped for trusted inputs)
    if (!skip_validation) {
        ValidateProgramSpec(spec, collected, metal_ctx, *mesh_device.allocator());
    }

    // Step 2: Build the program
    return BuildProgram(mesh_device, spec, collected, metal_ctx);
}

// ============================================================================
// Public Entry Points
// ============================================================================

Program MakeProgramFromSpec(distributed::MeshDevice& mesh_device, const ProgramSpec& spec, bool skip_validation) {
    Program program = BuildProgramFromSpec(mesh_device, spec, skip_validation);
    program.impl().compile_and_allocate(&mesh_device, false);
    return program;
}

distributed::MeshWorkload MakeMeshWorkloadFromSpecs(
    distributed::MeshDevice& mesh_device,
    const std::unordered_map<distributed::MeshCoordinateRange, ProgramSpec>& program_specs,
    bool skip_validation) {
    const distributed::MeshCoordinateRange mesh_extent(mesh_device.shape());
    distributed::MeshWorkload workload;
    TT_FATAL(!program_specs.empty(), "At least one ProgramSpec is required to create a MeshWorkload.");
    for (const auto& [device_range, program_spec] : program_specs) {
        TT_FATAL(
            mesh_extent.contains(device_range),
            "Device range {} is outside MeshDevice shape {}",
            device_range,
            mesh_device.shape());
        workload.impl().add_program(device_range, BuildProgramFromSpec(mesh_device, program_spec, skip_validation));
    }
    workload.impl().compile(&mesh_device);
    return workload;
}

distributed::MeshWorkload MakeMeshWorkloadFromSpec(
    distributed::MeshDevice& mesh_device, const ProgramSpec& program_spec, bool skip_validation) {
    distributed::MeshWorkload workload;
    workload.impl().add_program(
        distributed::MeshCoordinateRange(mesh_device.shape()),
        BuildProgramFromSpec(mesh_device, program_spec, skip_validation));
    workload.impl().compile(&mesh_device);
    return workload;
}

}  // namespace tt::tt_metal::experimental
