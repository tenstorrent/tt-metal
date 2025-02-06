
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/strong_type.hpp>

// Define common types used across TT-Mesh data-structures and APIs

using MeshTraceId = tt::stl::StrongType<uint32_t, struct MeshTraceIdTag>;

// TODO (Issue #17477): MeshWorkload and MeshEvent currently rely on the coordinate systems
// exposed below. These must be uplifted to an ND coordinate system (DeviceCoord and DeviceRange),
// keeping things more consistent  across the stack.
// For now, since the LogicalDeviceRange concept is fundamentally identical to the CoreRange concept
// on a 2D Mesh use this definition. CoreRange contains several utility functions required
// in the MeshWorkload context.

using DeviceCoord = CoreCoord;
using LogicalDeviceRange = CoreRange;
