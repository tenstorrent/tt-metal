// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/fabric/compute_mesh_router_builder.hpp"

namespace tt::tt_fabric {

// For now, switch mesh uses the same implementation as compute mesh.
// This will be replaced with a dedicated implementation when switch mesh
// hardware is available and we understand the specific requirements.
using SwitchMeshRouterBuilder = ComputeMeshRouterBuilder;

}  // namespace tt::tt_fabric
