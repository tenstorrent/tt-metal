// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <host_api.hpp>
#include <stdint.h>
#include <optional>
#include <vector>
#include <algorithm>

#include "core_coord.hpp"
#include "mesh_coord.hpp"
#include "sub_device_types.hpp"
#include "tt_metal/impl/program/dispatch.hpp"

namespace tt::tt_metal {
class IDevice;
class SystemMemoryManager;
}  // namespace tt::tt_metal

// Utility functions for dispatch MeshWorkloads
// Used by MeshCommandQueue
namespace tt::tt_metal::distributed {

// Translates a parent mesh coordinate range to submesh local coordinates.
// Returns nullopt if the range doesn't intersect with the submesh.
// The offset represents the submesh's origin in the parent mesh's coordinate system.
inline std::optional<MeshCoordinateRange> translate_range_by_offset(
    const MeshCoordinateRange& parent_range, const MeshCoordinate& offset, const MeshShape& submesh_shape) {
    if (parent_range.start_coord().dims() != offset.dims() || offset.dims() != submesh_shape.dims()) {
        return std::nullopt;
    }

    std::vector<uint32_t> translated_start(offset.dims());
    std::vector<uint32_t> translated_end(offset.dims());

    for (size_t i = 0; i < offset.dims(); i++) {
        // Check if range is entirely before or after the submesh
        if (parent_range.end_coord()[i] < offset[i] || parent_range.start_coord()[i] >= offset[i] + submesh_shape[i]) {
            return std::nullopt;  // No intersection
        }

        // Clamp to submesh bounds and translate to local coordinates
        uint32_t clamped_start = std::max(parent_range.start_coord()[i], offset[i]);
        uint32_t clamped_end = std::min(parent_range.end_coord()[i], offset[i] + submesh_shape[i] - 1);
        translated_start[i] = clamped_start - offset[i];
        translated_end[i] = clamped_end - offset[i];
    }

    return MeshCoordinateRange(MeshCoordinate(translated_start), MeshCoordinate(translated_end));
}

void write_go_signal(
    uint8_t cq_id,
    IDevice* device,
    SubDeviceId sub_device_id,
    SystemMemoryManager& sysmem_manager,
    uint32_t expected_num_workers_completed,
    CoreCoord dispatch_core,
    bool send_mcast,
    bool send_unicasts,
    const program_dispatch::ProgramDispatchMetadata& dispatch_md);

}  // namespace tt::tt_metal::distributed
