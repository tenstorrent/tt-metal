// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/program.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_coord.hpp>

#include <tt-metalium/workload/specification/program_spec.hpp>
#include "impl/metal2_host_api/program_spec/collection/collect_metadata.hpp"

#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {

// Builds the Program of `spec` on `mesh_device`.
// `device`: the coordinate of the one device that the Program runs on, if the caller knows it.
Program BuildProgram(
    distributed::MeshDevice& mesh_device,
    const std::optional<distributed::MeshCoordinate>& device,
    const ProgramSpec& spec,
    const CollectedSpecData& collected,
    MetalContext& metal_ctx);

}  // namespace tt::tt_metal
