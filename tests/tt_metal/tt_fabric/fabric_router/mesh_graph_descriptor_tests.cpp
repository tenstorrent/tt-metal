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

        top_level_instance: { mesh: { mesh_descriptor: "M0" id: 0 } }
    )proto";

    tt::tt_fabric::MeshGraphDescriptor desc(text_proto);

    EXPECT_THROW(tt::tt_fabric::MeshGraphDescriptor desc(text_proto, false), std::runtime_error);
}

TEST(MeshGraphDescriptorTests, ValidExpressConnections) {
    const std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1, 3 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 3 ] }
          express_connections: { src: 0 dst: 1 }
          express_connections: { src: 1 dst: 2 }
        }

        top_level_instance: { mesh: { mesh_descriptor: "M0" id: 0 } }
    )proto";

    // Should not throw - valid express connections within bounds (0-2 for 1x3 topology)
    tt::tt_fabric::MeshGraphDescriptor desc(text_proto);
}

TEST(MeshGraphDescriptorTests, ValidDeviceTopologyTypes) {
    const std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: {
            dims: [ 1, 4 ]
            types: [ LINE, RING ]
          }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 4 ] }
        }

        top_level_instance: { mesh: { mesh_descriptor: "M0" id: 0 } }
    )proto";

    // Should not throw - types array matches dimensions array size
    tt::tt_fabric::MeshGraphDescriptor desc(text_proto);
}

TEST(MeshGraphDescriptorTests, ValidEmptyDeviceTopologyTypes) {
    const std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: {
            dims: [ 1, 4 ]
            # types field omitted - should be valid
          }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 4 ] }
        }

        top_level_instance: { mesh: { mesh_descriptor: "M0" id: 0 } }
    )proto";

    // Should not throw - empty types array is valid
    tt::tt_fabric::MeshGraphDescriptor desc(text_proto);
}

TEST(MeshGraphDescriptorTests, ParsesFromTextProtoFile) {
    const std::filesystem::path text_proto_file_path =
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/mgd2_syntax_check_mesh_graph_descriptor.textproto";
    tt::tt_fabric::MeshGraphDescriptor desc(text_proto_file_path);
}

TEST(MeshGraphDescriptorTests, InvalidProtoStaticValidation) {
    // No mesh descriptors
    std::string text_proto = R"proto(
        top_level_instance: { mesh: { mesh_descriptor: "M0" id: 0 } }
    )proto";

    EXPECT_THROW(tt::tt_fabric::MeshGraphDescriptor desc(text_proto), std::runtime_error);

    // Missing mesh descriptor name
    text_proto = R"proto(
        mesh_descriptors: {
          # missing name
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1, 4 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 5 ] }
        }

        top_level_instance: { mesh: { mesh_descriptor: "M0" id: 0 } }
    )proto";

    EXPECT_THROW(tt::tt_fabric::MeshGraphDescriptor desc(text_proto), std::runtime_error);

    // Wrong number of dimensions
    text_proto = R"proto(
        mesh_descriptors: { name: "M0" arch: WORMHOLE_B0
                            device_topology: { dims: [ 1, 4, 3 ] }
    )proto";

    EXPECT_THROW(tt::tt_fabric::MeshGraphDescriptor desc(text_proto), std::runtime_error);

    // Device topology types size mismatch with dimensions
    text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: {
            dims: [ 1, 4 ]
            types: [ LINE ]
          }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 4 ] }
        }

        top_level_instance: { mesh: { mesh_descriptor: "M0" id: 0 } }
    )proto";

    EXPECT_THROW(tt::tt_fabric::MeshGraphDescriptor desc(text_proto), std::runtime_error);

    text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1 ] }
        }

        top_level_instance: { mesh: { mesh_descriptor: "M0" id: 0 } }
    )proto";

    EXPECT_THROW(tt::tt_fabric::MeshGraphDescriptor desc(text_proto), std::runtime_error);

    // Express connection out of bounds
    text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1, 3 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 3 ] }
          express_connections: { src: 5 dst: 1 }
        }

        top_level_instance: { mesh: { mesh_descriptor: "M0" id: 0 } }
    )proto";

    EXPECT_THROW(tt::tt_fabric::MeshGraphDescriptor desc(text_proto), std::runtime_error);
}

TEST(MeshGraphDescriptorTests, ValidProto) {}

TEST(MeshGraphDescriptorTests, InvalidProtoInstanceValidation) {
    std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1, 3 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 3 ] }
        }

        top_level_instance: { mesh: { mesh_descriptor: "M1" id: 0 } }
    )proto";

    // FIXME: Not done yet
    // EXPECT_THROW(tt::tt_fabric::MeshGraphDescriptor desc(text_proto), std::runtime_error);
}

}  // namespace fabric_router_tests

}  // namespace tt::tt_fabric
