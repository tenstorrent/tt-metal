// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_metal::distributed {

// Returns the set of ranges that result from subtracting the intersection from the parent range.
MeshCoordinateRangeSet subtract(const MeshCoordinateRange& parent, const MeshCoordinateRange& intersection);

// Refines partitions so each range is either fully contained in or disjoint from partitioning_range.
void partition_mesh_coordinate_ranges(
    std::vector<MeshCoordinateRange>& partitions, const MeshCoordinateRange& partitioning_range);

}  // namespace tt::tt_metal::distributed
