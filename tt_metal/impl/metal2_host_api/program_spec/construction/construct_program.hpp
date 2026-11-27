// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/workload/program.hpp>
#include <tt-metalium/mesh/legacy/device.hpp>
#include <tt-metalium/vocabulary/mesh_coord.hpp>

#include <tt-metalium/workload/specification/program_spec.hpp>
#include "impl/metal2_host_api/program_spec/collection/collect_metadata.hpp"

#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {

// Builds the Program of `spec` that runs on `devices` of `mesh_device`.
Program BuildProgram(
    distributed::MeshDevice& mesh_device,
    const distributed::MeshCoordinateRange& devices,
    const ProgramSpec& spec,
    const CollectedSpecData& collected,
    MetalContext& metal_ctx);

}  // namespace tt::tt_metal
