// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/fabric/fabric_types.hpp>

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"

namespace tt {
class Cluster;
}

namespace tt::tt_metal {
class Hal;
}

namespace tt::tt_fabric {

class ControlPlane;

// Fills in what only ControlPlane, the cluster, and the chip's other routers know.
manifest::Chip join_chip(
    manifest::Chip chip,
    const ControlPlane& control_plane,
    const tt::Cluster& cluster,
    FabricType fabric_type,
    FabricNodeId node,
    ChipId physical_chip_id);

// The L1 areas the HAL fixes on every router core.
manifest::Arch describe_arch(const tt::tt_metal::Hal& hal);

}  // namespace tt::tt_fabric
