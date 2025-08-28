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

// (Old message verification helpers removed.)

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

        top_level_instance: { mesh: { mesh_descriptor: "M0" id: 0 } }
    )proto";

    EXPECT_NO_THROW(MeshGraphDescriptor desc(text_proto));
}

TEST(MeshGraphDescriptorTests, ParsesFromTextProtoFile) {
    const std::filesystem::path text_proto_file_path =
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/mgd2_syntax_check_mesh_graph_descriptor.textproto";
    EXPECT_NO_THROW(MeshGraphDescriptor desc(text_proto_file_path));
}

TEST(MeshGraphDescriptorTests, InvalidProtoNoMeshDescriptors) {
    std::string text_proto = R"proto(
        top_level_instance: { mesh: { mesh_descriptor: "M0" id: 0 } }
    )proto";

    // Capture stdout, trigger validation, then print captured output and exception message
    try {
        MeshGraphDescriptor desc(text_proto);
        FAIL() << "Expected std::runtime_error for no mesh descriptors";
    } catch (const std::runtime_error& e) {
        // Verify new messages
        EXPECT_THAT(e.what(), ::testing::HasSubstr("Failed to validate MeshGraphDescriptor textproto"));
        EXPECT_THAT(e.what(), ::testing::HasSubstr("There must be at least one mesh descriptor"));
    }
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

    // Capture stdout, trigger validation, then print captured output and exception message
    try {
        MeshGraphDescriptor desc(text_proto);
        FAIL() << "Expected std::runtime_error for multiple validation failures";
    } catch (const std::runtime_error& e) {
        // Verify new messages
        EXPECT_THAT(e.what(), ::testing::HasSubstr("Failed to validate MeshGraphDescriptor textproto"));
        EXPECT_THAT(e.what(), ::testing::HasSubstr("Mesh descriptor 1 has no name"));
        EXPECT_THAT(e.what(), ::testing::HasSubstr("Device and host topology dimensions must be the same size"));
        EXPECT_THAT(e.what(), ::testing::HasSubstr("Architecture devices allow a maximum of 2 dimensions, but 3 were provided"));
    }
}

TEST(MeshGraphDescriptorTests, InvalidProtoMeshDescriptorValidation) {
    std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1, 3 ] }  # 3 devices total (0,1,2)
          channels: { count: 1 }
          host_topology: { dims: [ 1, 3 ] }
          express_connections: { src: 5 dst: 1 }   # src out of bounds
          express_connections: { src: 1 dst: 5 }   # dst out of bounds
        }

        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ -11, 3 ] }  # 3 devices total (0,1,2)
          channels: { count: 1 }
        }

        mesh_descriptors: {
          name: "M1"
          express_connections: { src: -1 dst: 1 }  # negative src
          express_connections: { src: 1 dst: -1 }  # negative dst
        }

        top_level_instance: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
    )proto";

    // Capture stdout, trigger validation, then print captured output and exception message
    try {
        MeshGraphDescriptor desc(text_proto);
        FAIL() << "Expected std::runtime_error for multiple express connection validation failures";
    } catch (const std::runtime_error& e) {
        // Verify new messages (subset)
        EXPECT_THAT(e.what(), ::testing::HasSubstr("Failed to validate MeshGraphDescriptor textproto"));
        EXPECT_THAT(e.what(), ::testing::HasSubstr("Mesh descriptor name is not unique (Mesh: M0)"));
        EXPECT_THAT(e.what(), ::testing::HasSubstr("Device topology dimensions must be positive (Mesh: M0)"));
        EXPECT_THAT(e.what(), ::testing::HasSubstr("Device and host topology dimensions must be the same size (Mesh: M0)"));
        EXPECT_THAT(e.what(), ::testing::HasSubstr("All mesh descriptors must have the same architecture"));
        EXPECT_THAT(e.what(), ::testing::HasSubstr("Express connection source is out of bounds (Mesh: M0)"));
        EXPECT_THAT(e.what(), ::testing::HasSubstr("Express connection destination is out of bounds (Mesh: M0)"));
    }
}

TEST(MeshGraphDescriptorTests, InvalidProtoGraphDescriptorValidation) {
    std::string text_proto = R"proto(
        graph_descriptors: {
            name: "G0"
            connections: {
                nodes: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
                nodes: { mesh: { mesh_descriptor: "M1" mesh_id: 0 } }
                channels: { count: -1 }
                directional: false
            }
            connections: {
                nodes: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
                channels: { count: 4 }
                directional: false
            }
        }

        graph_descriptors: {
            name: "G0"
            instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
            instances: { mesh: { mesh_descriptor: "M1" mesh_id: 0 } }
            graph_topology: {
                layout_type: ALL_TO_ALL
                channels: { count: -1 }
            }
        }

        graph_descriptors: {
            name: "G1"
            instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
            instances: { mesh: { mesh_descriptor: "M1" mesh_id: 0 } }
        }

        top_level_instance: { graph: { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    // Capture stdout, trigger validation, then print captured output and exception message
    try {
        MeshGraphDescriptor desc(text_proto);
        FAIL() << "Expected std::runtime_error for multiple validation failures";
    } catch (const std::runtime_error& e) {
        // Verify new messages (subset)
        EXPECT_THAT(e.what(), ::testing::HasSubstr("Failed to validate MeshGraphDescriptor textproto"));
        EXPECT_THAT(e.what(), ::testing::HasSubstr("There must be at least one mesh descriptor"));
        EXPECT_THAT(e.what(), ::testing::HasSubstr("Graph topology channel count must be positive (Graph: G0)"));
        EXPECT_THAT(e.what(), ::testing::HasSubstr("Connection channel count must be positive (Graph: G0)"));
        EXPECT_THAT(e.what(), ::testing::HasSubstr("Graph descriptor must have at least one instance (Graph: G0)"));
        EXPECT_THAT(e.what(), ::testing::HasSubstr("Graph descriptor must have a type specified (Graph: G1)"));
        EXPECT_THAT(e.what(), ::testing::HasSubstr("Connection must have at least two nodes (Graph: G0)"));
        EXPECT_THAT(e.what(), ::testing::HasSubstr("Graph descriptor must have either graph_topology or connections defined (Graph: G1)"));
    }
}



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
    // EXPECT_THROW(MeshGraphDescriptor desc(text_proto), std::runtime_error);
}

}  // namespace tt::tt_fabric::fabric_router_tests
