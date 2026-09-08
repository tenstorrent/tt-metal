// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>

namespace ttml::ttnn_fixed::distributed {

// Get the MGD (Mesh Graph Descriptor) path based on num_devices.
// Returns the path from TT_MESH_GRAPH_DESC_PATH env var if set,
// otherwise returns a default path based on num_devices (8 or 32).
// Also sets the TT_MESH_GRAPH_DESC_PATH env var if not already set.
// Returns std::nullopt if num_devices is not 8 or 32 and env var is not set.
std::optional<std::string> get_mgd_path(uint32_t num_devices);

// Select the mesh graph descriptor and fabric config to use for the next device open.
// Nothing is applied to the runtime here: core::MeshDevice picks both up when it builds
// its MetalEnv, because a MetalEnv reads TT_MESH_GRAPH_DESC_PATH and its fabric topology
// once, at construction.
void enable_fabric(uint32_t num_devices);

// Forget the fabric config selected by enable_fabric(), so the next device open runs
// without fabric. Without this, fabric stays armed for the remainder of the process and
// any subsequent default 1x1 open on a host where mmio_chip_ids().size() !=
// all_chip_ids().size() trips the "Fabric is being used but Device i is not active" check
// in tt_metal/impl/device/device_manager.cpp.
void disable_fabric();

// Fabric config chosen by the most recent enable_fabric(), or nullopt when fabric is off.
[[nodiscard]] std::optional<tt::tt_fabric::FabricConfig> selected_fabric_config();

}  // namespace ttml::ttnn_fixed::distributed
