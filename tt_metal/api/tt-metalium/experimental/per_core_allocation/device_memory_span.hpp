// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <unordered_map>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_metal::experimental::per_core_allocation {

// One contiguous span of caller-owned device memory on a node (SRAM only for now).
struct NodeLocalSpan {
    DeviceAddr address;
    uint64_t size;  // Bytes available starting at address, including any tensor padding.
};

// One span per participating core on a single device; this is not an allocation inventory.
struct PerNodeDeviceLocalMemorySpan {
    std::unordered_map<CoreCoord, NodeLocalSpan> per_node_spans;
};

// Spans on selected local devices; callers retain all backing storage.
struct PerNodeMeshDeviceMemorySpan {
    std::unordered_map<distributed::MeshCoordinate, PerNodeDeviceLocalMemorySpan> per_device_spans;
};

}  // namespace tt::tt_metal::experimental::per_core_allocation
