// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_map>
#include <cstdint>

namespace tt::tt_metal::distributed {

class MeshDevice;

// Manages allocator IDs and dependencies across a parent mesh and its submeshes.
// For now: single allocator state across all meshes; all AllocatorIDs are 0.
class SubmeshManager {
public:
    SubmeshManager() = default;

private:
};

}  // namespace tt::tt_metal::distributed
