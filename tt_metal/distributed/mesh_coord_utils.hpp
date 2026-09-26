// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_metal::distributed {

// Returns the set of ranges that result from subtracting the intersection from the parent range.
MeshCoordinateRangeSet subtract(const MeshCoordinateRange& parent, const MeshCoordinateRange& intersection);

}  // namespace tt::tt_metal::distributed
