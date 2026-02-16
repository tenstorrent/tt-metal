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

const char* kTTMetalHomeEnvVar = "TT_METAL_HOME";
const char* kTTMeshGraphDescriptorEnvVar = "TT_MESH_GRAPH_DESC_PATH";

namespace {

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

void enable_fabric(uint32_t num_devices) {
    std::string mesh_graph_descriptor_path;

    const char* mesh_graph_descriptor_path_env = std::getenv(kTTMeshGraphDescriptorEnvVar);
    if (mesh_graph_descriptor_path_env) {
        mesh_graph_descriptor_path = mesh_graph_descriptor_path_env;
    } else {
        const char* metal_home = std::getenv(kTTMetalHomeEnvVar);
        if (!metal_home) {
            throw std::runtime_error("TT_METAL_HOME is not set");
        }

        mesh_graph_descriptor_path = std::string(metal_home) + "/tests/tt_metal/tt_fabric/custom_mesh_descriptors/";

        bool set_env_var = (num_devices == 8U || num_devices == 32U);
        if (num_devices == 8U) {
            mesh_graph_descriptor_path += "t3k_1x8_mesh_graph_descriptor.textproto";
        } else if (num_devices == 32U) {
            mesh_graph_descriptor_path += "galaxy_1x32_mesh_graph_descriptor.textproto";
        }

        // set environment variable
        if (set_env_var) {
            setenv(kTTMeshGraphDescriptorEnvVar, mesh_graph_descriptor_path.c_str(), 1);
            fmt::println("[tt-train] TT_MESH_GRAPH_DESC_PATH is set to {}", mesh_graph_descriptor_path);
        }
    }

    // Infer the fabric config from the MGD's dim_types (LINE vs RING per axis)
    // This automatically selects FABRIC_2D, FABRIC_2D_TORUS_X, FABRIC_2D_TORUS_Y, or FABRIC_2D_TORUS_XY
    auto fabric_config = infer_fabric_config_from_mgd(mesh_graph_descriptor_path);
    tt::tt_fabric::SetFabricConfig(fabric_config);
}

}  // namespace ttml::ttnn_fixed::distributed
