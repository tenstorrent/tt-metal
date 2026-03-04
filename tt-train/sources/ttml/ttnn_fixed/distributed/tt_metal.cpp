// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal.hpp"

#include <fmt/core.h>

#include <core/ttnn_all_includes.hpp>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>

namespace ttml::ttnn_fixed::distributed {

namespace {

const char* kTTMetalRuntimeRootEnvVar = "TT_METAL_RUNTIME_ROOT";
const char* kTTMeshGraphDescriptorEnvVar = "TT_MESH_GRAPH_DESC_PATH";

// Convert FabricType (inferred from MGD dim_types) to the appropriate 2D FabricConfig
tt::tt_fabric::FabricConfig get_2d_fabric_config_from_type(tt::tt_fabric::FabricType fabric_type) {
    switch (fabric_type) {
        case tt::tt_fabric::FabricType::TORUS_X: return tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_X;
        case tt::tt_fabric::FabricType::TORUS_Y: return tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_Y;
        case tt::tt_fabric::FabricType::TORUS_XY: return tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_XY;
        case tt::tt_fabric::FabricType::MESH:
        default: return tt::tt_fabric::FabricConfig::FABRIC_2D;
    }
}

// Infer the appropriate FabricConfig from the mesh graph descriptor's dim_types
// Returns FABRIC_2D, FABRIC_2D_TORUS_X, FABRIC_2D_TORUS_Y, or FABRIC_2D_TORUS_XY
// based on whether each axis is LINE or RING in the MGD
tt::tt_fabric::FabricConfig infer_fabric_config_from_mgd(const std::string& mgd_path) {
    if (!std::filesystem::exists(mgd_path)) {
        fmt::println("[tt-train] MGD file not found: {}, using default FABRIC_2D", mgd_path);
        return tt::tt_fabric::FabricConfig::FABRIC_2D;
    }

    try {
        tt::tt_fabric::MeshGraphDescriptor mgd(std::filesystem::path(mgd_path), true /* backwards_compatible */);
        const auto& top_level = mgd.top_level();

        if (mgd.is_mesh(top_level)) {
            const auto* mesh_desc = std::get<const tt::tt_fabric::proto::MeshDescriptor*>(top_level.desc);
            auto inferred_fabric_type = tt::tt_fabric::MeshGraphDescriptor::infer_fabric_type_from_dim_types(mesh_desc);
            auto inferred_fabric_config = get_2d_fabric_config_from_type(inferred_fabric_type);

            fmt::println(
                "[tt-train] Inferred fabric config {} from MGD dim_types", static_cast<int>(inferred_fabric_config));
            return inferred_fabric_config;
        }
    } catch (const std::exception& e) {
        fmt::println("[tt-train] Failed to infer fabric config from MGD: {}, using default FABRIC_2D", e.what());
    }

    return tt::tt_fabric::FabricConfig::FABRIC_2D;
}

}  // namespace

std::optional<std::string> get_mgd_path(uint32_t num_devices) {
    // Check if env var is already set
    const char* mgd_path_env = std::getenv(kTTMeshGraphDescriptorEnvVar);
    if (mgd_path_env) {
        return std::string(mgd_path_env);
    }

    // Build default path based on num_devices
    const char* runtime_root = std::getenv(kTTMetalRuntimeRootEnvVar);
    if (!runtime_root) {
        throw std::runtime_error("TT_METAL_RUNTIME_ROOT is not set");
    }

    std::string mgd_path = std::string(runtime_root) + "/tt_metal/fabric/mesh_graph_descriptors/";

    if (num_devices == 8U) {
        mgd_path += "t3k_mesh_graph_descriptor.textproto";
    } else if (num_devices == 32U) {
        mgd_path += "single_galaxy_mesh_graph_descriptor.textproto";
    } else {
        // No default MGD for this device count
        return std::nullopt;
    }

    // Set environment variable for downstream use
    setenv(kTTMeshGraphDescriptorEnvVar, mgd_path.c_str(), 1);
    fmt::println("[tt-train] TT_MESH_GRAPH_DESC_PATH is set to {}", mgd_path);

    return mgd_path;
}

void enable_fabric(uint32_t num_devices) {
    auto mgd_path = get_mgd_path(num_devices);

    if (mgd_path.has_value()) {
        // Infer the fabric config from the MGD's dim_types (LINE vs RING per axis)
        // This automatically selects FABRIC_2D, FABRIC_2D_TORUS_X, FABRIC_2D_TORUS_Y, or FABRIC_2D_TORUS_XY
        auto fabric_config = infer_fabric_config_from_mgd(mgd_path.value());
        tt::tt_fabric::SetFabricConfig(fabric_config);
    } else {
        // No MGD available, use default FABRIC_2D
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_2D);
    }
}

}  // namespace ttml::ttnn_fixed::distributed
