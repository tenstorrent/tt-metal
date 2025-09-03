// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <filesystem>
#include <vector>
#include <string>
#include <cstdio>

#include <tt-metalium/mesh_graph_descriptor.hpp>

namespace tt::tt_fabric::fabric_router_tests {

TEST(MeshGraphDescriptorTests, ParsesFromTextProtoString) {
    const std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          dfsdadf: 3  # Allowing unknown fields for backwards compatibility
          arch: WORMHOLE_B0
          device_topology: {
            dims: [ 1, 4 ]
            dim_types: [ LINE, RING ]
          }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 4 ] }
          express_connections: { src: 0 dst: 1 }
          express_connections: { src: 1 dst: 2 }
        }

        top_level_instance: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
    )proto";

    EXPECT_NO_THROW(MeshGraphDescriptor desc(text_proto));
}

TEST(MeshGraphDescriptorTests, ParsesFromTextProtoFile) {
    const std::filesystem::path text_proto_file_path =
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/mgd2_syntax_check_mesh_graph_descriptor.textproto";
    // Sample file should parse successfully; unknown fields are allowed.
    EXPECT_NO_THROW(MeshGraphDescriptor desc(text_proto_file_path));
}

TEST(MeshGraphDescriptorTests, InvalidProtoNoMeshDescriptors) {
    std::string text_proto = R"proto(
        top_level_instance: { mesh: { mesh_descriptor: "M0" id: 0 } }
    )proto";

    EXPECT_THAT(
        ([&]() { MeshGraphDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::AllOf(
                ::testing::HasSubstr("Failed to validate MeshGraphDescriptor textproto"),
                ::testing::HasSubstr("There must be at least one mesh descriptor"))));
}

TEST(MeshGraphDescriptorTests, InvalidProtoDimensionValidationFailures) {
    std::string text_proto = R"proto(
        mesh_descriptors: {
          # missing name - will cause protobuf parsing issues
          arch: WORMHOLE_B0
          device_topology: {
            dims: [ 1, 4, 3 ]  # 3D for WORMHOLE_B0 (max 2D)
            dim_types: [ LINE ]  # size mismatch with dims
          }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 2 ] }  # dimension mismatch with device topology
        }

        top_level_instance: { mesh: { mesh_descriptor: "M0" id: 0 } }
    )proto";

    EXPECT_THAT(
        ([&]() { MeshGraphDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::AllOf(
                ::testing::HasSubstr("Failed to validate MeshGraphDescriptor textproto"),
                ::testing::HasSubstr("Mesh descriptor 1 has no name"))));
}

TEST(MeshGraphDescriptorTests, InvalidProtoArchitectureDimLimit) {
    std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1, 4, 3 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 4, 3 ] }
        }

        top_level_instance: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
    )proto";

    EXPECT_THAT(
        ([&]() { MeshGraphDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::AllOf(
                ::testing::HasSubstr("Failed to validate MeshGraphDescriptor textproto"),
                ::testing::HasSubstr("Architecture devices allow a maximum of 2 dimensions, but 3 were provided (Mesh: M0)"))));
}

TEST(MeshGraphDescriptorTests, InvalidProtoDeviceHostDimSizeMismatch) {
    std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1, 4 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1 ] }
        }

        top_level_instance: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
    )proto";

    EXPECT_THAT(
        ([&]() { MeshGraphDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::AllOf(
                ::testing::HasSubstr("Failed to validate MeshGraphDescriptor textproto"),
                ::testing::HasSubstr("Device and host topology dimensions must be the same size (Mesh: M0)"))));
}

TEST(MeshGraphDescriptorTests, InvalidProtoMixedArchitectures) {
    std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1, 2 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 2 ] }
        }

        mesh_descriptors: {
          name: "M1"
          arch: BLACKHOLE
          device_topology: { dims: [ 1, 2 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 2 ] }
        }

        top_level_instance: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
    )proto";

    EXPECT_THAT(
        ([&]() { MeshGraphDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::AllOf(
                ::testing::HasSubstr("Failed to validate MeshGraphDescriptor textproto"),
                ::testing::HasSubstr("All mesh descriptors must have the same architecture"))));
}

TEST(MeshGraphDescriptorTests, InvalidProtoExpressConnectionBounds) {
    std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1, 3 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 3 ] }
          express_connections: { src: 5 dst: 1 }
          express_connections: { src: 1 dst: 5 }
        }

        top_level_instance: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
    )proto";

    EXPECT_THAT(
        ([&]() { MeshGraphDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::AllOf(
                ::testing::HasSubstr("Failed to validate MeshGraphDescriptor textproto"),
                ::testing::HasSubstr("Express connection source is out of bounds (Mesh: M0)"),
                ::testing::HasSubstr("Express connection destination is out of bounds (Mesh: M0)"))));
}

TEST(MeshGraphDescriptorTests, InvalidGraphTopologyChannelCount) {
    std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1, 2 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 2 ] }
        }

        graph_descriptors: {
            name: "G0"
            type: "fabric"
            instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
            graph_topology: {
                layout_type: ALL_TO_ALL
                channels: { count: -1 }
            }
        }

        top_level_instance: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
    )proto";

    EXPECT_THAT(
        ([&]() { MeshGraphDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::AllOf(
                ::testing::HasSubstr("Failed to validate MeshGraphDescriptor textproto"),
                ::testing::HasSubstr("Graph topology channel count must be positive (Graph: G0)"))));
}

TEST(MeshGraphDescriptorTests, InvalidConnectionChannelCount) {
    std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1, 2 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 2 ] }
        }

        graph_descriptors: {
            name: "G0"
            type: "fabric"
            instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
            connections: {
                nodes: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
                nodes: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
                channels: { count: -1 }
                directional: false
            }
        }

        top_level_instance: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
    )proto";

    EXPECT_THAT(
        ([&]() { MeshGraphDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::AllOf(
                ::testing::HasSubstr("Failed to validate MeshGraphDescriptor textproto"),
                ::testing::HasSubstr("Connection channel count must be positive (Graph: G0)"))));
}

TEST(MeshGraphDescriptorTests, GraphMustHaveAtLeastOneInstance) {
    std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1, 2 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 2 ] }
        }

        graph_descriptors: {
            name: "G0"
            type: "fabric"
            connections: {
                nodes: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
                nodes: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
                channels: { count: 1 }
                directional: false
            }
        }

        top_level_instance: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
    )proto";

    EXPECT_THAT(
        ([&]() { MeshGraphDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::AllOf(
                ::testing::HasSubstr("Failed to validate MeshGraphDescriptor textproto"),
                ::testing::HasSubstr("Graph descriptor must have at least one instance (Graph: G0)"))));
}

TEST(MeshGraphDescriptorTests, GraphMustHaveTypeSpecified) {
    std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1, 2 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 2 ] }
        }

        graph_descriptors: {
            name: "G1"
            instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
            graph_topology: {
                layout_type: ALL_TO_ALL
                channels: { count: 1 }
            }
        }

        top_level_instance: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
    )proto";

    EXPECT_THAT(
        ([&]() { MeshGraphDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::AllOf(
                ::testing::HasSubstr("Failed to validate MeshGraphDescriptor textproto"),
                ::testing::HasSubstr("Graph descriptor must have a type specified (Graph: G1)"))));
}

TEST(MeshGraphDescriptorTests, ConnectionMustHaveAtLeastTwoNodes) {
    std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1, 2 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 2 ] }
        }

        graph_descriptors: {
            name: "G0"
            type: "fabric"
            instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
            connections: {
                nodes: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
                channels: { count: 1 }
                directional: false
            }
        }

        top_level_instance: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
    )proto";

    EXPECT_THAT(
        ([&]() { MeshGraphDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::AllOf(
                ::testing::HasSubstr("Failed to validate MeshGraphDescriptor textproto"),
                ::testing::HasSubstr("Connection must have at least two nodes (Graph: G0)"))));
}

TEST(MeshGraphDescriptorTests, GraphMustHaveTopologyOrConnections) {
    std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1, 2 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 2 ] }
        }

        graph_descriptors: {
            name: "G1"
            type: "fabric"
            instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
        }

        top_level_instance: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
    )proto";

    EXPECT_THAT(
        ([&]() { MeshGraphDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::AllOf(
                ::testing::HasSubstr("Failed to validate MeshGraphDescriptor textproto"),
                ::testing::HasSubstr("Graph descriptor must have either graph_topology or connections defined (Graph: G1)"))));
}

}  // namespace tt::tt_fabric::fabric_router_tests
