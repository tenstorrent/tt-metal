// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/mesh_config.hpp>

namespace tt::tt_metal::distributed {

// Factory class for creating DistributedMeshConfig instances
// Separates the concern of fetching mesh information from creating the config
class DistributedMeshConfigFactory {
public:
    // Create a DistributedMeshConfig from the current control plane state
    static DistributedMeshConfig create_from_control_plane(const MeshDeviceConfig& config);
};

}  // namespace tt::tt_metal::distributed