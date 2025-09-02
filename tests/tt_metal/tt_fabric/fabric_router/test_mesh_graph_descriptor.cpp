// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <tt-metalium/mesh_graph_descriptor.hpp>

namespace tt::tt_fabric {

namespace fabric_router_tests {

TEST(MeshGraphDescriptorTests, ParsesFromTextProtoString) {
    const std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          dfsdadf: 3  # Allowing unknown fields for backwards compatibility
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1, 4 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 5 ] }
        }

        top_level_instance: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
    )proto";

    tt::tt_fabric::MeshGraphDescriptor desc(text_proto);
}

TEST(MeshGraphDescriptorTests, ParsesFromTextProtoFile) {
    const std::filesystem::path text_proto_file_path =
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/mgd2_syntax_check_mesh_graph_descriptor.textproto";
    tt::tt_fabric::MeshGraphDescriptor desc(text_proto_file_path);
}

}  // namespace fabric_router_tests

}  // namespace tt::tt_fabric
