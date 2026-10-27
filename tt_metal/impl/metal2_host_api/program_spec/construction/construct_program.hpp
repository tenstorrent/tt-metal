// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program.hpp>
#include <tt-metalium/device.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "impl/metal2_host_api/program_spec/collection/collect_metadata.hpp"

#include "impl/context/metal_context.hpp"

namespace tt::tt_metal::experimental {

Program BuildProgram(
    distributed::MeshDevice& mesh_device,
    const ProgramSpec& spec,
    const CollectedSpecData& collected,
    MetalContext& metal_ctx);

}  // namespace tt::tt_metal::experimental
