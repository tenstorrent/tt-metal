// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal.hpp"

#include <fmt/core.h>

#include <core/ttnn_all_includes.hpp>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>

namespace ttml::ttnn_fixed::distributed {

const char* kTTMetalHomeEnvVar = "TT_METAL_HOME";
const char* kTTMeshGraphDescriptorEnvVar = "TT_MESH_GRAPH_DESC_PATH";

namespace {

const char* kFabricConfigEnvVar = "TT_TRAIN_FABRIC_CONFIG";

// Parse fabric config from environment variable string
// Supported values: "FABRIC_2D", "FABRIC_2D_TORUS_X", "FABRIC_2D_TORUS_Y", "FABRIC_2D_TORUS_XY"
std::optional<tt::tt_fabric::FabricConfig> parse_fabric_config_env() {
    const char* env = std::getenv(kFabricConfigEnvVar);
    if (!env) {
        return std::nullopt;
    }
    std::string config_str(env);
    if (config_str == "FABRIC_2D") {
        return tt::tt_fabric::FabricConfig::FABRIC_2D;
    } else if (config_str == "FABRIC_2D_TORUS_X") {
        return tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_X;
    } else if (config_str == "FABRIC_2D_TORUS_Y") {
        return tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_Y;
    } else if (config_str == "FABRIC_2D_TORUS_XY") {
        return tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_XY;
    }
    fmt::println("[tt-train] Unknown {} value: '{}', using default FABRIC_2D", kFabricConfigEnvVar, config_str);
    return std::nullopt;
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

    // Default to FABRIC_2D. TORUS configs can be explicitly opted-in via TT_TRAIN_FABRIC_CONFIG env var.
    // TORUS routing (FABRIC_2D_TORUS_X/Y/XY) enables wrap-around paths with dateline-based deadlock
    // avoidance, which changes routing behavior and can cause hangs with socket-based operations
    // (e.g. ring_shift) that don't fully support TORUS flow control.
    auto fabric_config = parse_fabric_config_env().value_or(tt::tt_fabric::FabricConfig::FABRIC_2D);
    fmt::println("[tt-train] Using fabric config: {}", static_cast<int>(fabric_config));
    tt::tt_fabric::SetFabricConfig(fabric_config);
}

}  // namespace ttml::ttnn_fixed::distributed
