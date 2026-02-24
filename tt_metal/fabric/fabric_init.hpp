// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device.hpp"
#include "tt_metal.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/distributed_context.hpp>
#include "llrt/tt_cluster.hpp"
#include "llrt/rtoptions.hpp"
#include "llrt/hal.hpp"

namespace tt::tt_fabric {

// Construct control plane using MGD
std::unique_ptr<tt::tt_fabric::ControlPlane> construct_control_plane(
    const std::filesystem::path& mesh_graph_desc_path,
    tt::Cluster& cluster,
    const ::tt::llrt::RunTimeOptions& rtoptions,
    const ::tt::tt_metal::Hal& hal,
    const tt_metal::distributed::multihost::DistributedContext& distributed_context,
    const tt_fabric::FabricConfig& fabric_config = tt_fabric::FabricConfig::DISABLED,
    const tt_fabric::FabricReliabilityMode& fabric_reliability_mode =
        tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE,
    const tt_fabric::FabricTensixConfig& fabric_tensix_config = tt_fabric::FabricTensixConfig::DISABLED,
    const tt_fabric::FabricUDMMode& fabric_udm_mode = tt_fabric::FabricUDMMode::DISABLED,
    const tt_fabric::FabricRouterConfig& fabric_router_config = tt_fabric::FabricRouterConfig{},
    const tt_fabric::FabricManagerMode& fabric_manager = tt_fabric::FabricManagerMode::DEFAULT,
    const std::map<tt_fabric::FabricNodeId, ChipId>& logical_mesh_chip_id_to_physical_chip_id_mapping = {});

// Construct control plane using auto-discovery
std::unique_ptr<tt::tt_fabric::ControlPlane> construct_control_plane(
    tt::Cluster& cluster,
    const ::tt::llrt::RunTimeOptions& rtoptions,
    const ::tt::tt_metal::Hal& hal,
    const tt_metal::distributed::multihost::DistributedContext& distributed_context,
    const tt_fabric::FabricConfig& fabric_config = tt_fabric::FabricConfig::DISABLED,
    const tt_fabric::FabricReliabilityMode& fabric_reliability_mode =
        tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE,
    const tt_fabric::FabricTensixConfig& fabric_tensix_config = tt_fabric::FabricTensixConfig::DISABLED,
    const tt_fabric::FabricUDMMode& fabric_udm_mode = tt_fabric::FabricUDMMode::DISABLED,
    const tt_fabric::FabricRouterConfig& fabric_router_config = tt_fabric::FabricRouterConfig{},
    const tt_fabric::FabricManagerMode& fabric_manager = tt_fabric::FabricManagerMode::DEFAULT,
    const std::map<tt_fabric::FabricNodeId, ChipId>& logical_mesh_chip_id_to_physical_chip_id_mapping = {});

// Compile fabric kernels needed to support scaleout systems.
std::unique_ptr<tt::tt_metal::Program> create_and_compile_fabric_program(tt::tt_metal::IDevice* device);

// Perform additional configuration (writing to specific L1 addresses, etc.) for fabric kernels on this device.
void configure_fabric_cores(tt::tt_metal::IDevice* device);

}  // namespace tt::tt_fabric
