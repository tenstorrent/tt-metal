// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <bit>
#include <functional>
#include <limits>
#include <numeric>
#include <map>
#include <set>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>  // HalMemType, for the borrowed-DFB per-bank sizing check
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>  // fmt::formatter<tt::DataFormat> for TT_FATAL messages
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <hostdevcommon/tensor_accessor/arg_config.hpp>
#include <tt_stl/fmt.hpp>
#include "impl/kernels/kernel.hpp"
#include "impl/metal2_host_api/llk_metadata.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/dispatch/dispatch_core_manager.hpp"
#include "impl/metal2_host_api/semaphore_scope.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "distributed/mesh_workload_impl.hpp"
#include <core_descriptor.hpp>
#include <llrt/tt_cluster.hpp>
#include <variant>

#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/metadata_collection/collect_metadata.hpp"
#include "impl/metal2_host_api/program_spec_validation/validate_spec.hpp"
#include "impl/metal2_host_api/program_construction/construct_program.hpp"

namespace tt::tt_metal::experimental {

Program BuildProgramFromSpec(distributed::MeshDevice& mesh_device, const ProgramSpec& spec, bool skip_validation) {
    log_debug(tt::LogMetal, "Creating Program from ProgramSpec ({})", spec.name);
    MetalContext& metal_ctx = mesh_device.impl().metal_context();

    // Step 1a: Collect derived data (builds lookup tables, checks structural invariants)
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
