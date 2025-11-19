// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal.hpp"

#include <fmt/core.h>

#include <core/ttnn_all_includes.hpp>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ttml::ttnn_fixed::distributed {

const char* kTTMetalHomeEnvVar = "TT_METAL_HOME";
const char* kTTMeshGraphDescriptorEnvVar = "TT_MESH_GRAPH_DESC_PATH";

void enable_fabric(uint32_t num_devices) {
    const char* mesh_graph_descriptor_path_env = std::getenv(kTTMeshGraphDescriptorEnvVar);
    if (!mesh_graph_descriptor_path_env) {
        const char* metal_home = std::getenv(kTTMetalHomeEnvVar);
        if (!metal_home) {
            throw std::runtime_error("TT_METAL_HOME is not set");
        }

        auto mesh_graph_descriptor_path =
            std::string(metal_home) + "/tests/tt_metal/tt_fabric/custom_mesh_descriptors/";

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

    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC);
}

}  // namespace ttml::ttnn_fixed::distributed
