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
#include <unordered_set>

#include <tt-metalium/mesh_graph_descriptor.hpp>

using namespace tt::tt_fabric;

// Helper functions for hierarchy testing
namespace {
void check_instance_count_by_type(const MeshGraphDescriptor& desc, const std::string& type, size_t expected_count) {
    const auto& ids = desc.instances_by_type(type);
    EXPECT_EQ(ids.size(), expected_count) << "Should have exactly " << expected_count << " " << type << " instances";
}

void check_instance_exists_by_name(const MeshGraphDescriptor& desc, const std::string& name, size_t expected_count = 1) {
    const auto& ids = desc.instances_by_name(name);
    EXPECT_EQ(ids.size(), expected_count) << "Should have exactly " << expected_count << " instance(s) with name '" << name << "'";
}

void check_instance_type(const MeshGraphDescriptor& desc, uint32_t global_id, bool should_be_graph) {
    const auto & inst = desc.get_instance(global_id);
    if (should_be_graph) {
        EXPECT_TRUE(desc.is_graph(inst)) << "Instance should be graph";
    } else {
        EXPECT_TRUE(desc.is_mesh(inst)) << "Instance should be mesh";
    }
}

std::set<std::string> get_instance_names_by_type(const MeshGraphDescriptor& desc, const std::string& type) {
    std::set<std::string> names;
    auto ids = desc.instances_by_type(type);
    for (uint32_t id : ids) {
        const auto & inst = desc.get_instance(id);
        names.insert(std::string(inst.name));
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
void check_sub_instances(const MeshGraphDescriptor& desc, const std::string& name, size_t expected_count, const std::unordered_set<std::string_view>& expected_names) {
    auto ids = desc.get_instance(desc.instances_by_name(name)[0]).sub_instances;
    EXPECT_EQ(ids.size(), expected_count) << "Should have exactly " << expected_count << " sub instances with name '" << name << "'";
    for (const auto & id : ids) {
        const auto & child = desc.get_instance(id);
        EXPECT_TRUE(expected_names.contains(child.name))
            << "Should have sub instance '" << child.name << "'";
    }
}

void expect_hierarchy_names(const MeshGraphDescriptor& desc, const std::string& instance_name, const std::vector<std::string>& expected_names) {
    const auto& ids = desc.instances_by_name(instance_name);
    ASSERT_FALSE(ids.empty()) << "No instance found with name '" << instance_name << "'";
    const auto & inst = desc.get_instance(ids[0]);
    std::vector<std::string> actual_names;
    actual_names.reserve(inst.hierarchy.size());
    for (auto nid : inst.hierarchy) {
        actual_names.emplace_back(std::string(desc.get_instance(nid).name));
    }
    EXPECT_EQ(actual_names, expected_names);
}

// Simple device checks for a mesh: only count and a few local IDs
void check_mesh_devices_simple(
    const MeshGraphDescriptor& desc,
    const std::string& mesh_name,
    size_t expected_devices,
    const std::vector<uint32_t>& sample_local_ids
) {
    const auto & mesh_ids = desc.instances_by_name(mesh_name);
    ASSERT_EQ(mesh_ids.size(), 1u) << "Expected exactly one instance named '" << mesh_name << "'";
    const auto & mesh_inst = desc.get_instance(mesh_ids[0]);
    ASSERT_TRUE(desc.is_mesh(mesh_inst)) << "'" << mesh_name << "' should be a mesh instance";

    EXPECT_EQ(mesh_inst.sub_instances.size(), expected_devices)
        << "Mesh '" << mesh_name << "' should have exactly " << expected_devices << " devices";

    for (auto local_id : sample_local_ids) {
        auto it = mesh_inst.sub_instances_local_id_to_global_id.find(local_id);
        ASSERT_TRUE(it != mesh_inst.sub_instances_local_id_to_global_id.end())
            << "Missing device local id " << local_id << " in mesh '" << mesh_name << "'";
        const auto & dev = desc.get_instance(it->second);
        EXPECT_EQ(dev.kind, NodeKind::Device);
        EXPECT_EQ(std::string(dev.type), "DEVICE");
        EXPECT_EQ(dev.local_id, local_id);
    }
}

void check_connections(
    MeshGraphDescriptor& desc,
    const std::vector<ConnectionId>& connections,
    const std::unordered_set<LocalNodeId>& expected_nodes,
    uint32_t expected_channel_count,
    bool expected_directional,
    GlobalNodeId expected_parent_instance_id,
    const std::unordered_set<std::string>& expected_node_names
) {
    for (size_t idx = 0; idx < connections.size(); ++idx) {
        const auto connection_id = connections[idx];
        const auto& connection = desc.get_connection(connection_id);

        EXPECT_EQ(connection.count, expected_channel_count);
        EXPECT_EQ(connection.directional, expected_directional);
        EXPECT_EQ(connection.parent_instance_id, expected_parent_instance_id);

        const auto& global_nodes = connection.nodes;


        auto dst_nodes = std::vector<GlobalNodeId>(global_nodes.begin() + 1, global_nodes.end());
        for (const auto& node : dst_nodes) {
            auto & instance = desc.get_instance(node);
            EXPECT_TRUE(expected_nodes.contains(instance.local_id))
                << "Connection " << connection_id << " should have node " << instance.local_id;
            EXPECT_TRUE(expected_node_names.contains(instance.name))
                << "Connection " << connection_id << " should have node " << instance.name;
        }
    }
}
}

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
    check_instance_count_by_type(desc, "MESH", 5);

    // Check specific instance names exist
    check_instances_have_names(desc, "CLUSTER", {"G2"});
    check_instances_have_names(desc, "POD", {"G0", "G1"});
    check_instances_have_names(desc, "MESH", {"M0", "M1", "M2", "M3", "M4"});

    // Check instance types (graph vs mesh)
    auto cluster_ids = desc.instances_by_type("CLUSTER");
    auto pod_ids = desc.instances_by_type("POD");
    auto mesh_ids = desc.instances_by_type("MESH");

    for (uint32_t id : cluster_ids) check_instance_type(desc, id, true);   // CLUSTER should be graph
    for (uint32_t id : pod_ids) check_instance_type(desc, id, true);       // POD should be graph
    for (uint32_t id : mesh_ids) check_instance_type(desc, id, false);     // mesh should be mesh

    // Check hierarchy relationships
    check_instance_exists_by_name(desc, "G0");
    check_instance_exists_by_name(desc, "G1");
    check_instance_exists_by_name(desc, "G2");

    // Check sub instance counts
    check_sub_instances(desc, "G0", 2, {"M0", "M1"});
    check_sub_instances(desc, "G1", 3, {"M3", "M2", "M4"});
    check_sub_instances(desc, "G2", 2, {"G1", "G0"});

    // Verify total instance count
    size_t total = desc.all_graphs().size() + desc.all_meshes().size();
    EXPECT_EQ(total, 8) << "Should have exactly 8 total instances (1 CLUSTER + 2 POD + 5 mesh)";

    // Check hierarchy chains
    expect_hierarchy_names(desc, "G2", {});
    expect_hierarchy_names(desc, "G0", {"G2"});
    expect_hierarchy_names(desc, "G1", {"G2"});
    expect_hierarchy_names(desc, "M0", {"G2", "G0"});
    expect_hierarchy_names(desc, "M1", {"G2", "G0"});
    expect_hierarchy_names(desc, "M2", {"G2", "G1"});
    expect_hierarchy_names(desc, "M3", {"G2", "G1"});
    expect_hierarchy_names(desc, "M4", {"G2", "G1"});


    // Simple device check for one mesh (M2): just count and a few local IDs
    check_mesh_devices_simple(desc, "M2", 8 * 4, {0, 5, 31});

    desc.print_all_nodes();
}

TEST(MeshGraphDescriptorTests, TestIntraMeshConnections) {
    // Single mesh, 2x3 devices, with a couple of valid express connections
    std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 2, 3 ]
                             dim_types: [ RING, LINE ] }
          channels: { count: 1 }
          host_topology: { dims: [ 2, 3 ] }
          express_connections: { src: 0 dst: 5 }
          express_connections: { src: 1 dst: 5 }
        }

        top_level_instance: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
    )proto";

    EXPECT_NO_THROW(MeshGraphDescriptor desc(text_proto));

    MeshGraphDescriptor desc(text_proto);

    desc.print_all_nodes();

    // Validate exactly one mesh instance and expected device population
    auto mesh_ids = desc.instances_by_type("MESH");
    ASSERT_EQ(mesh_ids.size(), 1);
    const auto & mesh_inst = desc.get_instance(mesh_ids[0]);
    ASSERT_TRUE(desc.is_mesh(mesh_inst));

    // 2x3 mesh => 6 devices; sample a few local IDs
    check_mesh_devices_simple(desc, "M0", 2 * 3, {0, 1, 2, 3, 5});

    // Check intra mesh connections
    const auto& all_connections = desc.connections_by_type("MESH");

    ASSERT_EQ(all_connections.size(), 24);

    // Layout should look like this with wrapping in x direction and express connections
    // 0 1 2
    // 3 4 5
    auto device_0 = desc.instances_by_name("D0")[0];
    auto connections = desc.connections_by_source_device_id(device_0);
    ASSERT_EQ(connections.size(), 4);
    check_connections(
        desc,
        connections,
        {1, 3, 5},
        1u,
        false,
        mesh_ids[0],
        {"D1","D3", "D5"}
    );

    auto device_1 = desc.instances_by_name("D1")[0];
    connections = desc.connections_by_source_device_id(device_1);
    ASSERT_EQ(connections.size(), 5);
    check_connections(
        desc,
        connections,
        {2, 4, 0, 5},
        1u,
        false,
        mesh_ids[0],
        {"D0", "D2", "D4", "D5"}
    );

    auto device_2 = desc.instances_by_name("D2")[0];
    connections = desc.connections_by_source_device_id(device_2);
    ASSERT_EQ(connections.size(), 3);
    check_connections(
        desc,
        connections,
        {1, 5},
        1u,
        false,
        mesh_ids[0],
        {"D1", "D5"}
    );
}


TEST(MeshGraphDescriptorTests, GraphInstancesWithDifferentGraphTypesError) {
    std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1, 1 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 1 ] }
        }

        # Two graph descriptors with different types
        graph_descriptors: {
          name: "G_POD_A"
          type: "POD"
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 1 } }
          graph_topology: { layout_type: ALL_TO_ALL channels: { count: 1 } }
        }

        graph_descriptors: {
          name: "G_POD_B"
          type: "PODX"
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 2 } }
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 3 } }
          graph_topology: { layout_type: ALL_TO_ALL channels: { count: 1 } }
        }

        # Cluster graph that mixes two POD graphs of different types
        graph_descriptors: {
          name: "G_CLUSTER"
          type: "CLUSTER"
          instances: { graph: { graph_descriptor: "G_POD_A" graph_id: 0 } }
          instances: { graph: { graph_descriptor: "G_POD_B" graph_id: 1 } }
          graph_topology: { layout_type: ALL_TO_ALL channels: { count: 1 } }
        }

        top_level_instance: { graph: { graph_descriptor: "G_CLUSTER" graph_id: 0 } }
    )proto";

    EXPECT_THAT(
        ([&]() { MeshGraphDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::AllOf(
                ::testing::HasSubstr("Graph instance type"),
                ::testing::HasSubstr("does not match graph descriptor child type"),
                ::testing::HasSubstr("POD"),
                ::testing::HasSubstr("PODX")
            )));
}

TEST(MeshGraphDescriptorTests, DuplicateInstanceIdsInGraphError) {
    std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1, 1 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 1 ] }
        }

        graph_descriptors: {
          name: "G_POD"
          type: "POD"
          # Duplicate mesh_id (0) for two instances
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
          graph_topology: { layout_type: ALL_TO_ALL channels: { count: 1 } }
        }

        top_level_instance: { graph: { graph_descriptor: "G_POD" graph_id: 0 } }
    )proto";

    EXPECT_THAT(
        ([&]() { MeshGraphDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::AllOf(
                ::testing::HasSubstr("Graph instance id"),
                ::testing::HasSubstr("already exists in this graph"),
                ::testing::HasSubstr("0")
            )));
}

TEST(MeshGraphDescriptorTests, MissingDescriptorReferencesInInstancesError) {
    // Case 1: Missing graph descriptor in top-level instance
    std::string text_proto_missing_graph = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1, 1 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 1 ] }
        }

        top_level_instance: { graph: { graph_descriptor: "G_MISSING" graph_id: 0 } }
    )proto";

    EXPECT_THAT(
        ([&]() { MeshGraphDescriptor desc(text_proto_missing_graph); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::AllOf(
                ::testing::HasSubstr("Graph descriptor G_MISSING not found in instance")
            )));

    // Case 2: Missing mesh descriptor referenced inside a graph descriptor instance
    std::string text_proto_missing_mesh = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1, 1 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 1 ] }
        }

        graph_descriptors: {
          name: "G_POD"
          type: "POD"
          instances: { mesh: { mesh_descriptor: "M_MISSING" mesh_id: 0 } }
          graph_topology: { layout_type: ALL_TO_ALL channels: { count: 1 } }
        }

        top_level_instance: { graph: { graph_descriptor: "G_POD" graph_id: 0 } }
    )proto";

    EXPECT_THAT(
        ([&]() { MeshGraphDescriptor desc(text_proto_missing_mesh); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::AllOf(
                ::testing::HasSubstr("Mesh descriptor M_MISSING not found in instance")
            )));
}

TEST(MeshGraphDescriptorTests, IntermeshConnectionsExplicitMultiLevelInvalid) {
    std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1, 2 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 1 ] }
        }

        graph_descriptors: {
          name: "G0"
          type: "POD"
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 1 } }
          connections: {
            nodes: {
              mesh: { mesh_descriptor: "M0" mesh_id: 0 } # << Mesh Level
            }
            nodes: {
              mesh: { mesh_descriptor: "M0" mesh_id: 1 device_id: 1 } # << One level down
            }
            channels: { count: 1 }
            directional: false
          }
        }
        top_level_instance: { graph: { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    EXPECT_THAT(
        ([&]() { MeshGraphDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::AllOf(
                ::testing::HasSubstr("Graph descriptor G0 connections must reference instances within same type")
            )));
}

TEST(MeshGraphDescriptorTests, IntermeshConnectionsExplicitMultiLevelInvalidChild) {
    std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1, 2 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 1 ] }
        }

        graph_descriptors: {
          name: "G0"
          type: "POD"
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 1 } }
          connections: {
            nodes: {
              graph: { graph_descriptor: "G0" graph_id: 0 } #<< These are wrong
            }
            nodes: {
              graph: { graph_descriptor: "G0" graph_id: 1} #<< These are wrong
            }
            channels: { count: 1 }
            directional: false
          }
        }
        top_level_instance: { graph: { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    EXPECT_THAT(
        ([&]() { MeshGraphDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::AllOf(
                ::testing::HasSubstr("Graph descriptor G0 does not match referenced instance M0")
            )));
}

TEST(MeshGraphDescriptorTests, IntermeshConnectionsExplicitMultiLevel) {
    std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1, 2 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 1 ] }
        }

        graph_descriptors: {
          name: "G0"
          type: "POD"
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 1 } }
          connections: {
            nodes: {
              mesh: { mesh_descriptor: "M0" mesh_id: 0 }
            }
            nodes: {
              mesh: { mesh_descriptor: "M0" mesh_id: 1 }
            }
            channels: { count: 1 }
            directional: false
          }
        }

        graph_descriptors: {
          name: "G1"
          type: "CLUSTER"
          instances: { graph: { graph_descriptor: "G0" graph_id: 0 } }
          instances: { graph: { graph_descriptor: "G0" graph_id: 1 } }

          # Explicit connections across multiple levels:
          # Connect device 0 in mesh M0(0) of G_POD(0) to device 1 in mesh M0(1) of G_POD(1)
          connections: {
            nodes: {
              graph: {
                graph_descriptor: "G0" graph_id: 0
                sub_ref: {
                  mesh: { mesh_descriptor: "M0" mesh_id: 0 device_id: 0 }
                }
              }
            }
            nodes: {
              graph: {
                graph_descriptor: "G0" graph_id: 1
                sub_ref: {
                  mesh: { mesh_descriptor: "M0" mesh_id: 1 device_id: 1 }
                }
              }
            }
            channels: { count: 2 }
            directional: false
          }
        }

        top_level_instance: { graph: { graph_descriptor: "G1" graph_id: 0 } }
    )proto";

    EXPECT_NO_THROW(MeshGraphDescriptor desc(text_proto));

    MeshGraphDescriptor desc(text_proto);

    {
        auto cluster_id = desc.instances_by_type("CLUSTER")[0];
        auto connections = desc.connections_by_instance_id(cluster_id);
        ASSERT_EQ(connections.size(), 2);
        check_connections(
            desc,
            connections,
            {0, 1},
            2u,
            false,
            cluster_id,
            {"D0", "D1"}
        );
    }
    {
        auto pod_ids = desc.instances_by_type("POD");

        for (auto pod_id : pod_ids) {
            auto connections = desc.connections_by_instance_id(pod_id);
            ASSERT_EQ(connections.size(), 2);
            check_connections(
                desc,
                connections,
                {0, 1},
                1u,
                false,
                pod_id,
                {"M0"}
            );
        }
    }
}

TEST(MeshGraphDescriptorTests, IntermeshConnectionsGraphTopologyAllToAll) {
    log_info(tt::LogTest, "NOTE: This test is skipped because topology connections are not yet implemented");
    GTEST_SKIP();
    // Topology shorthand case: two POD graphs, each containing two meshes.
    // The CLUSTER graph uses graph_topology: ALL_TO_ALL with channels.
    std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1, 2 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 1 ] }
        }

        graph_descriptors: {
          name: "G_POD"
          type: "POD"
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 1 } }
          graph_topology: { layout_type: ALL_TO_ALL channels: { count: 1 } }
        }

        graph_descriptors: {
          name: "G_CLUSTER"
          type: "CLUSTER"
          instances: { graph: { graph_descriptor: "G_POD" graph_id: 0 } }
          instances: { graph: { graph_descriptor: "G_POD" graph_id: 1 } }
          graph_topology: { layout_type: ALL_TO_ALL channels: { count: 2 } }
        }

        top_level_instance: { graph: { graph_descriptor: "G_CLUSTER" graph_id: 0 } }
    )proto";

    // Parsing and defaults should succeed
    EXPECT_NO_THROW(MeshGraphDescriptor desc(text_proto));

    MeshGraphDescriptor desc(text_proto);

    {
        auto cluster_id = desc.instances_by_type("CLUSTER")[0];
        auto connections = desc.connections_by_instance_id(cluster_id);
        ASSERT_EQ(connections.size(), 2);
        check_connections(
            desc,
            connections,
            {0, 1},
            2u,
            false,
            cluster_id,
            {"D0", "D1"}
        );
    }
    {
        auto pod_ids = desc.instances_by_type("POD");

        for (auto pod_id : pod_ids) {
            auto connections = desc.connections_by_instance_id(pod_id);
            ASSERT_EQ(connections.size(), 2);
            check_connections(
                desc,
                connections,
                {0, 1},
                1u,
                false,
                pod_id,
                {"M0"}
            );
        }
    }
}

TEST(MeshGraphDescriptorTests, DuplicateGraphDescriptorTypeInHierarchyError) {
    // Parent and child graphs share the same type (POD) which should be rejected
    std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1, 1 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 1 ] }
        }

        graph_descriptors: {
          name: "G_POD_CHILD"
          type: "POD"
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
          graph_topology: { layout_type: ALL_TO_ALL channels: { count: 1 } }
        }

        graph_descriptors: {
          name: "G_POD_PARENT"
          type: "POD"
          instances: { graph: { graph_descriptor: "G_POD_CHILD" graph_id: 0 } }
          graph_topology: { layout_type: ALL_TO_ALL channels: { count: 1 } }
        }

        top_level_instance: { graph: { graph_descriptor: "G_POD_PARENT" graph_id: 0 } }
    )proto";

    EXPECT_THAT(
        ([&]() { MeshGraphDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::AllOf(
                ::testing::HasSubstr("Graph descriptor type"),
                ::testing::HasSubstr("already exists in hierarchy"),
                ::testing::HasSubstr("POD")
            )));
}

// TODO: Test directional connections

}  // namespace tt::tt_fabric::fabric_router_tests
