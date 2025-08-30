// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <filesystem>
#include <vector>
#include <string>
#include <cstdio>
#include <set>

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

// Helper functions for hierarchy testing
namespace {
    void check_instance_count_by_type(const MeshGraphDescriptor& desc, const std::string& type, size_t expected_count) {
        auto ids = desc.get_ids_by_type(type);
        EXPECT_EQ(ids.size(), expected_count) << "Should have exactly " << expected_count << " " << type << " instances";
    }
    
    void check_instance_exists_by_name(const MeshGraphDescriptor& desc, const std::string& name, size_t expected_count = 1) {
        auto ids = desc.get_ids_by_name(name);
        EXPECT_EQ(ids.size(), expected_count) << "Should have exactly " << expected_count << " instance(s) with name '" << name << "'";
    }
    
    void check_instance_type(const MeshGraphDescriptor& desc, uint32_t global_id, bool should_be_graph) {
        auto* instance = desc.get_instance_by_global_id(global_id);
        EXPECT_NE(instance, nullptr) << "Instance with global ID " << global_id << " should exist";
        if (instance) {
            if (should_be_graph) {
                EXPECT_TRUE(std::holds_alternative<const proto::GraphDescriptor*>(instance->descriptor))
                    << "Instance should have graph descriptor";
            } else {
                EXPECT_TRUE(std::holds_alternative<const proto::MeshDescriptor*>(instance->descriptor))
                    << "Instance should have mesh descriptor";
            }
        }
    }
    
    std::set<std::string> get_instance_names_by_type(const MeshGraphDescriptor& desc, const std::string& type) {
        std::set<std::string> names;
        auto ids = desc.get_ids_by_type(type);
        for (uint32_t id : ids) {
            auto* instance = desc.get_instance_by_global_id(id);
            if (instance) {
                names.insert(instance->descriptor_name);
            }
        }
        return names;
    }
    
    void check_instances_have_names(const MeshGraphDescriptor& desc, const std::string& type, const std::vector<std::string>& expected_names) {
        auto names = get_instance_names_by_type(desc, type);
        for (const auto& expected_name : expected_names) {
            EXPECT_TRUE(names.find(expected_name) != names.end()) 
                << "Should have " << type << " instance '" << expected_name << "'";
        }
    }
}

TEST(MeshGraphDescriptorTests, TestInstanceCreation) {
    const std::filesystem::path text_proto_file_path =
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/mgd2_syntax_check_mesh_graph_descriptor.textproto";

    // Sample file should parse successfully; unknown fields are allowed.
    EXPECT_NO_THROW(MeshGraphDescriptor desc(text_proto_file_path));

    // Test instance creation and hierarchy
    MeshGraphDescriptor desc(text_proto_file_path);
    
    // Check hierarchy levels with helper functions
    check_instance_count_by_type(desc, "CLUSTER", 1);
    check_instance_count_by_type(desc, "POD", 2);
    check_instance_count_by_type(desc, "mesh", 5);
    
    // Check specific instance names exist
    check_instances_have_names(desc, "CLUSTER", {"G2"});
    check_instances_have_names(desc, "POD", {"G0", "G1"});
    check_instances_have_names(desc, "mesh", {"M0", "M1", "M2", "M3", "M4"});
    
    // Check instance types (graph vs mesh)
    auto cluster_ids = desc.get_ids_by_type("CLUSTER");
    auto pod_ids = desc.get_ids_by_type("POD");
    auto mesh_ids = desc.get_ids_by_type("mesh");
    
    for (uint32_t id : cluster_ids) check_instance_type(desc, id, true);   // CLUSTER should be graph
    for (uint32_t id : pod_ids) check_instance_type(desc, id, true);       // POD should be graph
    for (uint32_t id : mesh_ids) check_instance_type(desc, id, false);     // mesh should be mesh
    
    // Check hierarchy relationships
    check_instance_exists_by_name(desc, "G0");
    check_instance_exists_by_name(desc, "G1");
    check_instance_exists_by_name(desc, "G2");
    
    // Verify total instance count
    auto all_ids = desc.get_all_ids();
    EXPECT_EQ(all_ids.size(), 8) << "Should have exactly 8 total instances (1 CLUSTER + 2 POD + 5 mesh)";
}

}  // namespace tt::tt_fabric::fabric_router_tests
