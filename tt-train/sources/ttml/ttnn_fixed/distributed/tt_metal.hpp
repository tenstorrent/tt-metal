// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace ttml::ttnn_fixed::distributed {

// Get the MGD (Mesh Graph Descriptor) path based on num_devices.
// Returns the path from TT_MESH_GRAPH_DESC_PATH env var if set,
// otherwise returns a default path based on num_devices (8 or 32).
// Also sets the TT_MESH_GRAPH_DESC_PATH env var if not already set.
// Returns std::nullopt if num_devices is not 8 or 32 and env var is not set.
std::optional<std::string> get_mgd_path(uint32_t num_devices);

void enable_fabric(uint32_t num_devices);

// Tear down any previously installed fabric config by transitioning to
// FabricConfig::DISABLED. Safe to call when no devices are open (per
// tt_metal/impl/context/metal_env.cpp: "Going TO DISABLED is allowed"),
// which is the case immediately after AutoContext::close_device() and
// during the failure path of open_device_mesh().
void disable_fabric();

}  // namespace ttml::ttnn_fixed::distributed
