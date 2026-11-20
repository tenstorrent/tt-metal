// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include <unordered_map>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt_stl/fmt.hpp>

#include "distributed/mesh_device_impl.hpp"
#include "distributed/mesh_workload_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/metal2_host_api/program_spec/collection/collect_metadata.hpp"
#include "impl/metal2_host_api/program_spec/construction/construct_program.hpp"
#include "impl/metal2_host_api/program_spec/validation/validate_spec.hpp"
#include "impl/program/program_impl.hpp"

namespace tt::tt_metal {

namespace {

// The coordinate of the device of `range`, if the range has exactly one device.
std::optional<distributed::MeshCoordinate> SingleDeviceOf(const distributed::MeshCoordinateRange& range) {
    if (range.shape().mesh_size() != 1) {
        return std::nullopt;
    }
    return range.start_coord();
}

// `spec`, after checking that the Program constructor can create its Program.
const ProgramSpec& RequireProgramConstructorSupport(const ProgramSpec& spec) {
    TT_FATAL(
        spec.fabric_connections.empty(),
        "ProgramSpec '{}' has fabric connections: only a MeshWorkload constructor can create its Program, because "
        "the connection args differ per device",
        spec.name);
    return spec;
}

}  // namespace

// `device`: the coordinate of the one device that the Program runs on, if the caller knows it.
Program BuildProgramFromSpec(
    distributed::MeshDevice& mesh_device,
    const std::optional<distributed::MeshCoordinate>& device,
    const ProgramSpec& spec) {
    log_debug(tt::LogMetal, "Creating Program from ProgramSpec ({})", spec.name);
    MetalContext& metal_ctx = mesh_device.impl().metal_context();

    // Step 1a: Collect derived data (builds lookup tables, checks name uniqueness and references)
    CollectedSpecData collected = CollectSpecData(spec);

    // Step 1b: Validate semantic rules
    ValidateProgramSpec(spec, collected, metal_ctx, *mesh_device.allocator());

    // Step 2: Build the program
    return BuildProgram(mesh_device, device, spec, collected, metal_ctx);
}

// ============================================================================
// Public Entry Points
// ============================================================================

Program::Program(distributed::MeshDevice& mesh_device, const ProgramSpec& spec) :
    Program(BuildProgramFromSpec(mesh_device, std::nullopt, RequireProgramConstructorSupport(spec))) {
    impl().compile_and_allocate(&mesh_device, false);
}

distributed::MeshWorkload::MeshWorkload(
    distributed::MeshDevice& mesh_device,
    const std::unordered_map<distributed::MeshCoordinateRange, ProgramSpec>& program_specs) :
    MeshWorkload() {
    const distributed::MeshCoordinateRange mesh_extent(mesh_device.shape());
    TT_FATAL(!program_specs.empty(), "At least one ProgramSpec is required to create a MeshWorkload.");
    for (const auto& [device_range, program_spec] : program_specs) {
        TT_FATAL(
            mesh_extent.contains(device_range),
            "Device range {} is outside MeshDevice shape {}",
            device_range,
            mesh_device.shape());
        impl().add_program(device_range, BuildProgramFromSpec(mesh_device, SingleDeviceOf(device_range), program_spec));
    }
    impl().compile(&mesh_device);
}

distributed::MeshWorkload::MeshWorkload(distributed::MeshDevice& mesh_device, const ProgramSpec& program_spec) :
    MeshWorkload() {
    const distributed::MeshCoordinateRange device_range(mesh_device.shape());
    impl().add_program(device_range, BuildProgramFromSpec(mesh_device, SingleDeviceOf(device_range), program_spec));
    impl().compile(&mesh_device);
}

}  // namespace tt::tt_metal
