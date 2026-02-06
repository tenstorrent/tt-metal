// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device.hpp"
#include "tt_metal.hpp"
#include <experimental/fabric/fabric_types.hpp>

namespace tt {
class Cluster;
}  // namespace tt

namespace tt::tt_fabric {

class ControlPlane;

// Compile fabric kernels needed to support scaleout systems.
std::unique_ptr<tt::tt_metal::Program> create_and_compile_fabric_program(
    tt::tt_metal::IDevice* device,
    FabricConfig fabric_config,
    ControlPlane& control_plane,
    const tt::Cluster& cluster,
    FabricTensixConfig fabric_tensix_config,
    bool fast_dispatch);

// Perform additional configuration (writing to specific L1 addresses, etc.) for fabric kernels on this device.
void configure_fabric_cores(tt::tt_metal::IDevice* device, const tt::Cluster& cluster, ControlPlane& control_plane);

}  // namespace tt::tt_fabric
