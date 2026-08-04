// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <filesystem>
#include <vector>
#include <string>
#include <cstdio>
#include <set>
#include <map>
#include <unordered_set>
#include <fstream>

#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>

#include "cluster.hpp"
#include "impl/context/metal_context.hpp"

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
        EXPECT_TRUE(names.contains(expected_name)) << "Should have " << type << " instance '" << expected_name << "'";
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
    GlobalNodeId expected_parent_instance_id,
    const std::unordered_set<std::string>& expected_node_names) {
    for (unsigned int connection_id : connections) {
        const auto& connection = desc.get_connection(connection_id);

        EXPECT_EQ(connection.count, expected_channel_count);
        EXPECT_EQ(connection.parent_instance_id, expected_parent_instance_id);

        const auto& global_nodes = connection.nodes;


        auto dst_nodes = std::vector<GlobalNodeId>(global_nodes.begin() + 1, global_nodes.end());
        for (const auto& node : dst_nodes) {
            const auto& instance = desc.get_instance(node);
            EXPECT_TRUE(expected_nodes.contains(instance.local_id))
                << "Connection " << connection_id << " should have node " << instance.local_id;
            EXPECT_TRUE(expected_node_names.contains(instance.name))
                << "Connection " << connection_id << " should have node " << instance.name;
        }
    }
}

// Mirror of internal::get_all_mgd_fabric_types(): pull each mesh instance's MeshDescriptor out
// of the parsed MGD and infer its FabricType from the wired dim_types, one entry per mesh.
std::vector<FabricType> infer_fabric_types(const MeshGraphDescriptor& desc) {
    std::vector<FabricType> types;
    for (const auto mesh : desc.all_meshes()) {
        const auto& instance = desc.get_instance(mesh);
        const auto* mesh_desc = std::get<const proto::MeshDescriptor*>(instance.desc);
        types.push_back(MeshGraphDescriptor::infer_fabric_type_from_dim_types(mesh_desc));
    }
    return types;
}

std::string single_mesh_proto(const std::string& dim_types) {
    return R"proto(
               mesh_descriptors: { name: "M0" arch: WORMHOLE_B0
                                   device_topology: { dims: [ 4, 4 ]
                                                      dim_types:)proto" +
           dim_types + R"proto(
        }
        channels: { count: 1 }
          host_topology: { dims: [ 1, 1 ] }
        }
        top_level_instance: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
    )proto";
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

TEST(MeshGraphDescriptorTests, InfersDeclaredTorusTypeForDegenerateDimensions) {
    const std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: {
            dims: [ 2, 4 ]
            dim_types: [ RING, RING ]
          }
          channels: { count: 1 }
          host_topology: { dims: [ 2, 4 ] }
        }
        top_level_instance: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
    )proto";

    MeshGraphDescriptor desc(text_proto);
    const auto& instance = desc.get_instance(desc.instances_by_name("M0").at(0));
    const auto* mesh_desc = std::get<const proto::MeshDescriptor*>(instance.desc);
    EXPECT_EQ(MeshGraphDescriptor::infer_fabric_type_from_dim_types(mesh_desc), FabricType::TORUS_XY);
}

TEST(MeshGraphDescriptorTests, InfersDeclaredTorusTypeForDegenerateSwitchDimensions) {
    const std::string text_proto = R"proto(
        switch_descriptors: {
          name: "SW0"
          arch: WORMHOLE_B0
          device_topology: {
            dims: [ 2, 4 ]
            dim_types: [ RING, RING ]
          }
          channels: { count: 1 }
        }
        top_level_instance: { switch: { switch_descriptor: "SW0" switch_id: 0 } }
    )proto";

    MeshGraphDescriptor desc(text_proto);
    const auto& instance = desc.get_instance(desc.instances_by_name("SW0").at(0));
    const auto* switch_desc = std::get<const proto::SwitchDescriptor*>(instance.desc);
    EXPECT_EQ(MeshGraphDescriptor::infer_fabric_type_from_dim_types(switch_desc), FabricType::TORUS_XY);
}

TEST(MeshGraphDescriptorTests, CollapsedTorusSwitchRetainsMeshDirectionsAndEdgePorts) {
    const std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1, 1 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 1 ] }
        }
        switch_descriptors: {
          name: "SW0"
          arch: WORMHOLE_B0
          device_topology: {
            dims: [ 2, 4 ]
            dim_types: [ RING, RING ]
          }
          channels: { count: 1 }
        }
        graph_descriptors: {
          name: "G0"
          type: "FABRIC"
          instances: { switch: { switch_descriptor: "SW0" switch_id: 0 } }
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 1 } }
        }
        top_level_instance: { graph: { graph_descriptor: "G0" graph_id: 0 } }
    )proto";
    const auto test_file = std::filesystem::temp_directory_path() / "test_collapsed_torus_switch.textproto";
    {
        std::ofstream file(test_file);
        file << text_proto;
    }

    tt::tt_fabric::MeshGraph mesh_graph(tt::tt_metal::ClusterType::T3K, test_file.string());
    std::filesystem::remove(test_file);

    const auto& connectivity = mesh_graph.get_intra_mesh_connectivity().at(0);
    EXPECT_EQ(connectivity.at(0).at(4).port_direction, RoutingDirection::S);
    EXPECT_EQ(connectivity.at(4).at(0).port_direction, RoutingDirection::N);

    const auto& edge_ports = mesh_graph.get_mesh_edge_ports_to_chip_id().at(0);
    EXPECT_EQ(edge_ports.at({RoutingDirection::N, 0}), 0);
    EXPECT_EQ(edge_ports.at({RoutingDirection::S, 0}), 4);
}

TEST(MeshGraphDescriptorTests, ParsesFromTextProtoFile) {
    const std::filesystem::path text_proto_file_path =
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/mgd2_syntax_check_mesh_graph_descriptor.textproto";
    // Sample file should parse successfully; unknown fields are allowed.
    EXPECT_NO_THROW(MeshGraphDescriptor desc(text_proto_file_path));
}

TEST(MeshGraphDescriptorTests, ParsesBhGalaxySp4TorusXY) {
    const std::filesystem::path text_proto_file_path =
        "tt_metal/fabric/mesh_graph_descriptors/bh_galaxy_sp4_torus_xy_graph_descriptor.textproto";
    EXPECT_NO_THROW(MeshGraphDescriptor desc(text_proto_file_path));
}

// Covers the dim_types -> FabricType inference behind get_all_mgd_fabric_types() (the Python API
// exposed in tt-blaze). dim_types are [Y, X]; a RING axis makes that axis a torus.
TEST(MeshGraphDescriptorTests, InferFabricTypeFromDimTypes) {
    struct Case {
        std::string dim_types;
        FabricType expected;
    };
    const std::vector<Case> cases = {
        {"[ LINE, LINE ]", FabricType::MESH},
        {"[ LINE, RING ]", FabricType::TORUS_X},
        {"[ RING, LINE ]", FabricType::TORUS_Y},
        {"[ RING, RING ]", FabricType::TORUS_XY},
    };
    for (const auto& c : cases) {
        MeshGraphDescriptor desc(single_mesh_proto(c.dim_types));
        const auto types = infer_fabric_types(desc);
        ASSERT_EQ(types.size(), 1u) << "dim_types " << c.dim_types;
        EXPECT_EQ(types[0], c.expected) << "dim_types " << c.dim_types;
    }
}

// One FabricType per compute mesh: a graph of two meshes with different topologies must yield
// both meshes' inferred types.
TEST(MeshGraphDescriptorTests, InferFabricTypePerMeshInMultiMeshGraph) {
    const std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M_TX"
          arch: WORMHOLE_B0
          device_topology: {
            dims: [ 4, 4 ]
            dim_types: [ LINE, RING ]
          }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 1 ] }
        }
        mesh_descriptors: {
          name: "M_TY"
          arch: WORMHOLE_B0
          device_topology: {
            dims: [ 4, 4 ]
            dim_types: [ RING, LINE ]
          }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 1 ] }
        }

        graph_descriptors: {
          name: "G0"
          type: "FABRIC"
          instances: { mesh: { mesh_descriptor: "M_TX" mesh_id: 0 } }
          instances: { mesh: { mesh_descriptor: "M_TY" mesh_id: 1 } }
          connections: {
            nodes: { mesh: { mesh_descriptor: "M_TX" mesh_id: 0 } }
            nodes: { mesh: { mesh_descriptor: "M_TY" mesh_id: 1 } }
            channels: { count: 1 }
          }
        }

        top_level_instance: { graph: { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    MeshGraphDescriptor desc(text_proto);
    const auto types = infer_fabric_types(desc);
    ASSERT_EQ(types.size(), 2u) << "one FabricType per compute mesh";
    const std::set<FabricType> got(types.begin(), types.end());
    EXPECT_EQ(got, (std::set<FabricType>{FabricType::TORUS_X, FabricType::TORUS_Y}));
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

    for (uint32_t id : cluster_ids) {
        check_instance_type(desc, id, true);  // CLUSTER should be graph
    }
    for (uint32_t id : pod_ids) {
        check_instance_type(desc, id, true);  // POD should be graph
    }
    for (uint32_t id : mesh_ids) {
        check_instance_type(desc, id, false);  // mesh should be mesh
    }

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

    // Test all_names() returns unique names (as unordered_set)
    {
        auto all_names = desc.all_names();
        // Verify it's a set (no duplicates) by checking size matches expected unique count
        EXPECT_GE(all_names.size(), 8u) << "Should have at least 8 unique names (G2, G0, G1, M0-M4)";

        // Verify expected names are present
        EXPECT_TRUE(all_names.contains("G2")) << "Should contain G2";
        EXPECT_TRUE(all_names.contains("G0")) << "Should contain G0";
        EXPECT_TRUE(all_names.contains("G1")) << "Should contain G1";
        EXPECT_TRUE(all_names.contains("M0")) << "Should contain M0";
        EXPECT_TRUE(all_names.contains("M1")) << "Should contain M1";
        EXPECT_TRUE(all_names.contains("M2")) << "Should contain M2";
        EXPECT_TRUE(all_names.contains("M3")) << "Should contain M3";
        EXPECT_TRUE(all_names.contains("M4")) << "Should contain M4";

        // Verify no duplicates by checking that inserting all names into a new set gives same size
        std::unordered_set<std::string> verification_set(all_names.begin(), all_names.end());
        EXPECT_EQ(all_names.size(), verification_set.size())
            << "all_names() should return unique names (no duplicates)";
    }

    // Test all_types() returns unique types (as unordered_set)
    {
        auto all_types = desc.all_types();
        // Verify it's a set (no duplicates)
        EXPECT_GE(all_types.size(), 4u) << "Should have at least 4 unique types (CLUSTER, POD, MESH, DEVICE)";

        // Verify expected types are present
        EXPECT_TRUE(all_types.contains("CLUSTER")) << "Should contain CLUSTER type";
        EXPECT_TRUE(all_types.contains("POD")) << "Should contain POD type";
        EXPECT_TRUE(all_types.contains("MESH")) << "Should contain MESH type";
        EXPECT_TRUE(all_types.contains("DEVICE")) << "Should contain DEVICE type";

        // Verify no duplicates by checking that inserting all types into a new set gives same size
        std::unordered_set<std::string> verification_set(all_types.begin(), all_types.end());
        EXPECT_EQ(all_types.size(), verification_set.size())
            << "all_types() should return unique types (no duplicates)";
    }

    desc.print_all_nodes();
}

TEST(MeshGraphDescriptorTests, TestGetChipCount) {
    // Test get_chip_count() with various mesh dimensions
    const std::filesystem::path text_proto_file_path =
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/mgd2_syntax_check_mesh_graph_descriptor.textproto";

    MeshGraphDescriptor desc(text_proto_file_path);

    // Get mesh instances
    auto mesh_ids = desc.instances_by_type("MESH");
    ASSERT_GE(mesh_ids.size(), 1u) << "Should have at least one mesh instance";

    // Test get_chip_count with GlobalNodeId
    for (auto mesh_id : mesh_ids) {
        const auto& mesh_instance = desc.get_instance(mesh_id);
        ASSERT_TRUE(desc.is_mesh(mesh_instance)) << "Instance should be a mesh";

        uint32_t chip_count = desc.get_chip_count(mesh_id);
        EXPECT_GT(chip_count, 0u) << "Chip count should be positive for mesh " << mesh_instance.name;

        // Test get_chip_count with InstanceData reference
        uint32_t chip_count_ref = desc.get_chip_count(mesh_instance);
        EXPECT_EQ(chip_count, chip_count_ref) << "Both overloads should return same chip count";
    }

    // Test with a specific mesh file that has known dimensions
    const std::filesystem::path dual_4x4_file_path =
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_4x4_mesh_graph_descriptor.textproto";

    if (std::filesystem::exists(dual_4x4_file_path)) {
        MeshGraphDescriptor dual_desc(dual_4x4_file_path);
        auto dual_mesh_ids = dual_desc.instances_by_type("MESH");

        for (auto mesh_id : dual_mesh_ids) {
            uint32_t chip_count = dual_desc.get_chip_count(mesh_id);
            // 4x4 mesh should have 16 chips
            EXPECT_EQ(chip_count, 16u) << "4x4 mesh should have 16 chips";
        }
    }

    // Test with 32x4 mesh if available
    const std::filesystem::path mesh_32x4_file_path =
        "tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto";

    if (std::filesystem::exists(mesh_32x4_file_path)) {
        MeshGraphDescriptor mesh_32x4_desc(mesh_32x4_file_path);
        auto mesh_32x4_ids = mesh_32x4_desc.instances_by_type("MESH");

        for (auto mesh_id : mesh_32x4_ids) {
            uint32_t chip_count = mesh_32x4_desc.get_chip_count(mesh_id);
            // 32x4 mesh should have 128 chips
            EXPECT_EQ(chip_count, 128u) << "32x4 mesh should have 128 chips";
        }
    }

    // Test error case: get_chip_count on non-mesh instance should fail
    // Note: TT_FATAL will abort, so we can't test this with EXPECT_THROW
    // This is expected behavior - the function should only be called on mesh instances
    auto graph_ids = desc.instances_by_type("POD");
    if (!graph_ids.empty()) {
        const auto& graph_instance = desc.get_instance(graph_ids[0]);
        EXPECT_FALSE(desc.is_mesh(graph_instance)) << "Graph instance should not be a mesh";
        // The function will TT_FATAL if called on non-mesh, which is correct behavior
    }
}

TEST(MeshGraphDescriptorTests, TestCountInstancesByType) {
    const std::filesystem::path text_proto_file_path =
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/mgd2_syntax_check_mesh_graph_descriptor.textproto";

    MeshGraphDescriptor desc(text_proto_file_path);

    // Test counting existing types
    {
        std::vector<std::string> types = {"MESH", "POD", "CLUSTER"};
        auto counts = desc.count_instances_by_type(types);

        EXPECT_EQ(counts.size(), 3u) << "Should return counts for all requested types";
        EXPECT_EQ(counts.at("MESH"), 5u) << "Should have 5 MESH instances";
        EXPECT_EQ(counts.at("POD"), 2u) << "Should have 2 POD instances";
        EXPECT_EQ(counts.at("CLUSTER"), 1u) << "Should have 1 CLUSTER instance";
    }

    // Test counting with non-existent types
    {
        std::vector<std::string> types = {"MESH", "NONEXISTENT", "POD"};
        auto counts = desc.count_instances_by_type(types);

        EXPECT_EQ(counts.size(), 3u) << "Should return counts for all requested types";
        EXPECT_EQ(counts.at("MESH"), 5u) << "Should have 5 MESH instances";
        EXPECT_EQ(counts.at("NONEXISTENT"), 0u) << "Non-existent type should return 0";
        EXPECT_EQ(counts.at("POD"), 2u) << "Should have 2 POD instances";
    }

    // Test with empty types list
    {
        std::vector<std::string> types = {};
        auto counts = desc.count_instances_by_type(types);

        EXPECT_EQ(counts.size(), 0u) << "Empty types list should return empty map";
    }

    // Test with single type
    {
        std::vector<std::string> types = {"MESH"};
        auto counts = desc.count_instances_by_type(types);

        EXPECT_EQ(counts.size(), 1u) << "Should return count for single type";
        EXPECT_EQ(counts.at("MESH"), 5u) << "Should have 5 MESH instances";
    }
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
    check_connections(desc, connections, {1, 3, 5}, 1u, mesh_ids[0], {"D1", "D3", "D5"});

    auto device_1 = desc.instances_by_name("D1")[0];
    connections = desc.connections_by_source_device_id(device_1);
    ASSERT_EQ(connections.size(), 5);
    check_connections(desc, connections, {2, 4, 0, 5}, 1u, mesh_ids[0], {"D0", "D2", "D4", "D5"});

    auto device_2 = desc.instances_by_name("D2")[0];
    connections = desc.connections_by_source_device_id(device_2);
    ASSERT_EQ(connections.size(), 3);
    check_connections(desc, connections, {1, 5}, 1u, mesh_ids[0], {"D1", "D5"});

    // Test all_names() returns unique names (as unordered_set)
    {
        auto all_names = desc.all_names();
        // Verify it's a set (no duplicates) - should have M0 + 6 devices = 7 unique names
        EXPECT_EQ(all_names.size(), 7u) << "Should have exactly 7 unique names (M0 + D0-D5)";

        // Verify expected names are present (M0 mesh and D0-D5 devices)
        EXPECT_TRUE(all_names.contains("M0")) << "Should contain M0";
        for (int i = 0; i < 6; ++i) {
            EXPECT_TRUE(all_names.contains("D" + std::to_string(i))) << "Should contain D" << i;
        }

        // Verify no duplicates by checking that inserting all names into a new set gives same size
        std::unordered_set<std::string> verification_set(all_names.begin(), all_names.end());
        EXPECT_EQ(all_names.size(), verification_set.size())
            << "all_names() should return unique names (no duplicates)";
    }

    // Test all_types() returns unique types (as unordered_set)
    {
        auto all_types = desc.all_types();
        // Verify it's a set (no duplicates) - should have exactly 2 types
        EXPECT_EQ(all_types.size(), 2u) << "Should have exactly 2 unique types (MESH, DEVICE)";

        // Verify expected types are present
        EXPECT_TRUE(all_types.contains("MESH")) << "Should contain MESH type";
        EXPECT_TRUE(all_types.contains("DEVICE")) << "Should contain DEVICE type";

        // Verify no duplicates by checking that inserting all types into a new set gives same size
        std::unordered_set<std::string> verification_set(all_types.begin(), all_types.end());
        EXPECT_EQ(all_types.size(), verification_set.size())
            << "all_types() should return unique types (no duplicates)";
    }
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
        check_connections(desc, connections, {0, 1}, 2u, cluster_id, {"D0", "D1"});
    }
    {
        auto pod_ids = desc.instances_by_type("POD");

        for (auto pod_id : pod_ids) {
            auto connections = desc.connections_by_instance_id(pod_id);
            ASSERT_EQ(connections.size(), 2);
            check_connections(desc, connections, {0, 1}, 1u, pod_id, {"M0"});
        }
    }
}

TEST(MeshGraphDescriptorTests, IntermeshConnectionsGraphTopologyAllToAll) {
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
          graph_topology: {
            layout_type: ALL_TO_ALL
            channels: { count: 1 }
          }
        }

        graph_descriptors: {
          name: "G_CLUSTER"
          type: "CLUSTER"
          instances: { graph: { graph_descriptor: "G_POD" graph_id: 0 } }
          instances: { graph: { graph_descriptor: "G_POD" graph_id: 1 } }
          graph_topology: {
            layout_type: ALL_TO_ALL
            channels: { count: 2 }
          }
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
        check_connections(desc, connections, {0, 1}, 2u, cluster_id, {"G_POD"});
    }
    {
        auto pod_ids = desc.instances_by_type("POD");

        for (auto pod_id : pod_ids) {
            auto connections = desc.connections_by_instance_id(pod_id);
            ASSERT_EQ(connections.size(), 2);
            check_connections(desc, connections, {0, 1}, 1u, pod_id, {"M0"});
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

TEST(MeshGraphDescriptorTests, AllToAllGraphTopology) {
    std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 2, 2 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 1 ] }
        }

        graph_descriptors: {
          name: "G0"
          type: "POD"
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 1 } }
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 2 } }
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 3 } }
          graph_topology: {
            layout_type: ALL_TO_ALL
            channels: { count: 1 }
          }
          connections: {
            # One extra explicit connection (bidirectional: directional inter-mesh connections are rejected,
            # see MeshGraphDescriptorTests.DirectionalConnectionsAreRejected and issue #50292).
            nodes: { mesh: { mesh_descriptor: "M0" mesh_id: 0 device_id: 0 } }
            nodes: { mesh: { mesh_descriptor: "M0" mesh_id: 1 device_id: 1 } }
            channels: { count: 1 }
            directional: false
          }
        }

        top_level_instance: { graph: { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    EXPECT_NO_THROW(MeshGraphDescriptor desc(text_proto));

    MeshGraphDescriptor desc(text_proto);

    auto pod_id = desc.instances_by_type("POD")[0];
    {
        auto connections = desc.connections_by_instance_id(pod_id);
        // 12 all-to-all mesh-level connections + 2 for the bidirectional explicit device-level connection.
        ASSERT_EQ(connections.size(), 14);
        check_connections(desc, connections, {0, 1, 2, 3}, 1u, pod_id, {"M0", "D1", "D0"});
    }
    // Check connections from M0(0)
    {
        auto pod_instance = desc.get_instance(pod_id);
        auto connections = desc.connections_by_source_device_id(pod_instance.sub_instances_local_id_to_global_id.at(0));
        ASSERT_EQ(connections.size(), 3);
        check_connections(desc, connections, {1, 2, 3}, 1u, pod_id, {"M0", "D1"});
    }
}

TEST(MeshGraphDescriptorTests, RingGraphTopology) {
    std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 2, 2 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 1 ] }
        }
        mesh_descriptors: {
          name: "M1"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 2, 2 ] }
          channels: { count: 2 }
          host_topology: { dims: [ 1, 1 ] }
        }

        graph_descriptors: {
          name: "G0"
          type: "POD"
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 1 } }
          instances: { mesh: { mesh_descriptor: "M1" mesh_id: 2 } }
          instances: { mesh: { mesh_descriptor: "M1" mesh_id: 3 } }
          graph_topology: {
            layout_type: RING
            channels: { count: 1 }
          }
        }

        top_level_instance: { graph: { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    EXPECT_NO_THROW(MeshGraphDescriptor desc(text_proto));

    MeshGraphDescriptor desc(text_proto);

    auto pod_id = desc.instances_by_type("POD")[0];
    {
        auto connections = desc.connections_by_instance_id(pod_id);
        ASSERT_EQ(connections.size(), 8);
        check_connections(desc, connections, {0, 1, 2, 3}, 1u, pod_id, {"M0", "M1"});
    }
    // Check connections from M0(0)
    {
        auto pod_instance = desc.get_instance(pod_id);
        auto connections = desc.connections_by_source_device_id(pod_instance.sub_instances_local_id_to_global_id.at(0));
        ASSERT_EQ(connections.size(), 2);
        check_connections(desc, connections, {1, 3}, 1u, pod_id, {"M0", "M1"});
    }
}

TEST(MeshGraphDescriptorTests, BidirectionalConnections) {
    // Test that when directional=false, connections exist in both directions
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
            nodes: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
            nodes: { mesh: { mesh_descriptor: "M0" mesh_id: 1 } }
            channels: { count: 2 }
            directional: false
          }
        }

        top_level_instance: { graph: { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    EXPECT_NO_THROW(MeshGraphDescriptor desc(text_proto));

    MeshGraphDescriptor desc(text_proto);

    auto pod_id = desc.instances_by_type("POD")[0];
    const auto& pod_instance = desc.get_instance(pod_id);
    auto mesh_0_device_0 = pod_instance.sub_instances_local_id_to_global_id.at(0);
    auto mesh_1_device_0 = pod_instance.sub_instances_local_id_to_global_id.at(1);

    // Check that both devices have outgoing connections (bidirectional)
    const auto& connections_from_mesh_0 = desc.connections_by_source_device_id(mesh_0_device_0);
    const auto& connections_from_mesh_1 = desc.connections_by_source_device_id(mesh_1_device_0);

    ASSERT_EQ(connections_from_mesh_0.size(), 1);
    ASSERT_EQ(connections_from_mesh_1.size(), 1);
}

TEST(MeshGraphDescriptorTests, DirectionalConnectionsAreRejected) {
    // Directional inter-mesh connections are not supported end-to-end (issue #50292): only the authored
    // direction is stored, so the peer endpoint never gathers the cable in the control plane and strict binding
    // resolves 0 routers. MeshGraphDescriptor now hard-fails at parse time instead of silently mis-configuring.
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
            nodes: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
            nodes: { mesh: { mesh_descriptor: "M0" mesh_id: 1 } }
            channels: { count: 3 }
            directional: true
          }
        }

        top_level_instance: { graph: { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    EXPECT_THAT(
        ([&]() { MeshGraphDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(::testing::AllOf(
            ::testing::HasSubstr("directional inter-mesh connection"),
            ::testing::HasSubstr("not fully supported"),
            ::testing::HasSubstr("50292"))));
}

TEST(MeshGraphDescriptorTests, ParsesSwitchDescriptor) {
    const std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 2, 4 ] }
          channels: { count: 2 }
          host_topology: { dims: [ 1, 1 ] }
        }

        switch_descriptors: {
          name: "SW0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 2, 4 ] }
          channels: { count: 2 }
        }

        graph_descriptors: {
          name: "G0"
          type: "FABRIC"
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
          instances: { switch: { switch_descriptor: "SW0" switch_id: 2 } }
          connections: {
            nodes: { mesh: { mesh_descriptor: "M0" mesh_id: 0 device_id: 2 } }
            nodes: { switch: { switch_descriptor: "SW0" switch_id: 2 device_id: 2 } }
            channels: { count: 2 }
          }
        }

        top_level_instance: { graph: { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    EXPECT_NO_THROW(MeshGraphDescriptor desc(text_proto));
}

TEST(MeshGraphDescriptorTests, SwitchInstanceCreation) {
    const std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 2, 4 ] }
          channels: { count: 2 }
          host_topology: { dims: [ 1, 1 ] }
        }

        switch_descriptors: {
          name: "SW0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 2, 4 ] }
          channels: { count: 2 }
        }

        graph_descriptors: {
          name: "G0"
          type: "FABRIC"
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 1 } }
          instances: { switch: { switch_descriptor: "SW0" switch_id: 2 } }
          connections: {
            nodes: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
            nodes: { switch: { switch_descriptor: "SW0" switch_id: 2 } }
            channels: { count: 2 }
          }
          connections: {
            nodes: { mesh: { mesh_descriptor: "M0" mesh_id: 1 } }
            nodes: { switch: { switch_descriptor: "SW0" switch_id: 2 } }
            channels: { count: 2 }
          }
        }

        top_level_instance: { graph: { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    MeshGraphDescriptor desc(text_proto);

    // Check that switch instances are created
    const auto& switch_instances = desc.all_switches();
    EXPECT_EQ(switch_instances.size(), 1) << "Should have exactly 1 switch instance";

    // Check switch instance properties
    const auto& switch_instance = desc.get_instance(switch_instances[0]);
    EXPECT_TRUE(desc.is_switch(switch_instance)) << "Instance should be a switch";
    EXPECT_EQ(std::string(switch_instance.name), "SW0") << "Switch should have name SW0";
    EXPECT_EQ(switch_instance.type, "SWITCH") << "Switch type should be SWITCH";
    EXPECT_EQ(switch_instance.local_id, 2) << "Switch should have local_id 2 (as specified in switch_id: 2)";

    // Check that switch has devices
    EXPECT_EQ(switch_instance.sub_instances.size(), 8) << "Switch should have 2*4=8 devices";

    // Check switch devices
    for (LocalNodeId i = 0; i < 8; ++i) {
        auto it = switch_instance.sub_instances_local_id_to_global_id.find(i);
        ASSERT_TRUE(it != switch_instance.sub_instances_local_id_to_global_id.end())
            << "Missing device local id " << i << " in switch";
        const auto& dev = desc.get_instance(it->second);
        EXPECT_EQ(dev.kind, NodeKind::Device);
        EXPECT_EQ(std::string(dev.type), "DEVICE");
        EXPECT_EQ(dev.local_id, i);
    }
}

TEST(MeshGraphDescriptorTests, SwitchConnections) {
    const std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 2, 4 ] }
          channels: { count: 2 }
          host_topology: { dims: [ 1, 1 ] }
        }

        switch_descriptors: {
          name: "SW0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 2, 4 ] }
          channels: { count: 2 }
        }

        graph_descriptors: {
          name: "G0"
          type: "FABRIC"
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 1 } }
          instances: { switch: { switch_descriptor: "SW0" switch_id: 2 } }
          connections: {
            nodes: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
            nodes: { switch: { switch_descriptor: "SW0" switch_id: 2 } }
            channels: { count: 2 }
          }
          connections: {
            nodes: { mesh: { mesh_descriptor: "M0" mesh_id: 1 } }
            nodes: { switch: { switch_descriptor: "SW0" switch_id: 2 } }
            channels: { count: 2 }
          }
        }

        top_level_instance: { graph: { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    MeshGraphDescriptor desc(text_proto);

    // Get switch instance
    const auto& switch_instances = desc.all_switches();
    ASSERT_EQ(switch_instances.size(), 1);
    const auto& switch_instance = desc.get_instance(switch_instances[0]);
    auto switch_device_2 = switch_instance.sub_instances_local_id_to_global_id.at(2);
    auto switch_device_3 = switch_instance.sub_instances_local_id_to_global_id.at(3);

    // Check connections from switch devices
    const auto& connections_from_switch_dev_2 = desc.connections_by_source_device_id(switch_device_2);
    const auto& connections_from_switch_dev_3 = desc.connections_by_source_device_id(switch_device_3);

    EXPECT_GT(connections_from_switch_dev_2.size(), 0) << "Switch device 2 should have connections";
    EXPECT_GT(connections_from_switch_dev_3.size(), 0) << "Switch device 3 should have connections";
}

TEST(MeshGraphDescriptorTests, SwitchValidationSingleHost) {
    // Test that switch descriptor without host_topology is valid (single host implicit)
    const std::string text_proto = R"proto(
        switch_descriptors: {
          name: "SW0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 2, 4 ] }
          channels: { count: 2 }
        }

        top_level_instance: { switch: { switch_descriptor: "SW0" switch_id: 0 } }
    )proto";

    EXPECT_NO_THROW(MeshGraphDescriptor desc(text_proto));
}

TEST(MeshGraphDescriptorTests, SwitchValidationInvalidDimensions) {
    // Test validation fails for invalid switch dimensions
    const std::string text_proto = R"proto(
        switch_descriptors: {
          name: "SW0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 0, 4 ] }  # Invalid: 0 dimension
          channels: { count: 2 }
        }

        top_level_instance: { switch: { switch_descriptor: "SW0" switch_id: 0 } }
    )proto";

    EXPECT_THAT(
        ([&]() { MeshGraphDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Failed to validate MeshGraphDescriptor textproto")));
}

TEST(MeshGraphDescriptorTests, SwitchValidationInvalidChannels) {
    // Test validation fails for invalid channel count
    const std::string text_proto = R"proto(
        switch_descriptors: {
          name: "SW0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 2, 4 ] }
          channels: { count: 0 }  # Invalid: 0 channels
        }

        top_level_instance: { switch: { switch_descriptor: "SW0" switch_id: 0 } }
    )proto";

    EXPECT_THAT(
        ([&]() { MeshGraphDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Failed to validate MeshGraphDescriptor textproto")));
}

TEST(MeshGraphDescriptorTests, SwitchExpressConnections) {
    // Test switch with express connections
    const std::string text_proto = R"proto(
        switch_descriptors: {
          name: "SW0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 2, 4 ] }
          channels: { count: 2 }
          express_connections: { src: 0 dst: 4 }
          express_connections: { src: 1 dst: 5 }
        }

        top_level_instance: { switch: { switch_descriptor: "SW0" switch_id: 0 } }
    )proto";

    EXPECT_NO_THROW(MeshGraphDescriptor desc(text_proto));
}

TEST(MeshGraphDescriptorTests, SwitchMixedWithMeshesInGraph) {
    // Test that switches can be mixed with meshes in a graph via explicit connections
    const std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 2, 4 ] }
          channels: { count: 2 }
          host_topology: { dims: [ 1, 1 ] }
        }

        switch_descriptors: {
          name: "SW0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 2, 4 ] }
          channels: { count: 2 }
        }

        graph_descriptors: {
          name: "G0"
          type: "FABRIC"
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 1 } }
          instances: { switch: { switch_descriptor: "SW0" switch_id: 2 } }
          connections: {
            nodes: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
            nodes: { switch: { switch_descriptor: "SW0" switch_id: 2 } }
            channels: { count: 2 }
          }
          connections: {
            nodes: { mesh: { mesh_descriptor: "M0" mesh_id: 1 } }
            nodes: { switch: { switch_descriptor: "SW0" switch_id: 2 } }
            channels: { count: 2 }
          }
        }

        top_level_instance: { graph: { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    EXPECT_NO_THROW(MeshGraphDescriptor desc(text_proto));

    MeshGraphDescriptor desc(text_proto);

    // Verify we have meshes and switches
    EXPECT_EQ(desc.all_meshes().size(), 2) << "Should have 2 mesh instances";
    EXPECT_EQ(desc.all_switches().size(), 1) << "Should have 1 switch instance";
}

TEST(MeshGraphDescriptorTests, AssignZDirectionInMeshGraph) {
    // Test that assign_z_direction flag is properly tracked in MeshGraph
    const std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1, 2 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 1 ] }
        }
        mesh_descriptors: {
          name: "M1"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1, 2 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 1 ] }
        }
        mesh_descriptors: {
          name: "M2"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 1, 2 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 1 ] }
        }

        graph_descriptors: {
          name: "G0"
          type: "FABRIC"
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
          instances: { mesh: { mesh_descriptor: "M1" mesh_id: 1 } }
          instances: { mesh: { mesh_descriptor: "M2" mesh_id: 2 } }
          connections: {
            nodes: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
            nodes: { mesh: { mesh_descriptor: "M1" mesh_id: 1 } }
            channels: { count: 2 }
            assign_z_direction: true
          }
          connections: {
            nodes: { mesh: { mesh_descriptor: "M0" mesh_id: 0 device_id: 0 } }
            nodes: { mesh: { mesh_descriptor: "M2" mesh_id: 2 device_id: 0 } }
            channels: { count: 1 }
            assign_z_direction: true
          }
          connections: {
            nodes: { mesh: { mesh_descriptor: "M1" mesh_id: 1 } }
            nodes: { mesh: { mesh_descriptor: "M2" mesh_id: 2 } }
            channels: { count: 2 }
            # assign_z_direction not specified, should default to false
          }
        }

        top_level_instance: { graph: { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    // Create a temporary file for the test
    const std::filesystem::path test_file =
        std::filesystem::temp_directory_path() / "test_assign_z_direction.textproto";
    {
        std::ofstream file(test_file);
        file << text_proto;
    }

    // Cluster type doesn't matter
    const tt::tt_metal::ClusterType cluster_type = tt::tt_metal::ClusterType::BLACKHOLE_GALAXY;
    EXPECT_NO_THROW(tt::tt_fabric::MeshGraph mesh_graph(cluster_type, test_file.string()));

    tt::tt_fabric::MeshGraph mesh_graph(cluster_type, test_file.string());

    // Test should_assign_z_direction method
    tt::tt_fabric::MeshId mesh_0(0);
    tt::tt_fabric::MeshId mesh_1(1);
    tt::tt_fabric::MeshId mesh_2(2);

    // M0 <-> M1 should use Z direction (mesh-level connection with assign_z_direction: true)
    EXPECT_TRUE(mesh_graph.should_assign_z_direction(mesh_0, mesh_1)) << "M0 <-> M1 should use Z direction";
    EXPECT_TRUE(mesh_graph.should_assign_z_direction(mesh_1, mesh_0))
        << "M1 <-> M0 should use Z direction (bidirectional)";

    // M0 <-> M2 should use Z direction (device-level connection with assign_z_direction: true)
    EXPECT_TRUE(mesh_graph.should_assign_z_direction(mesh_0, mesh_2)) << "M0 <-> M2 should use Z direction";
    EXPECT_TRUE(mesh_graph.should_assign_z_direction(mesh_2, mesh_0))
        << "M2 <-> M0 should use Z direction (bidirectional)";

    // M1 <-> M2 should NOT use Z direction (assign_z_direction not specified, defaults to false)
    EXPECT_FALSE(mesh_graph.should_assign_z_direction(mesh_1, mesh_2)) << "M1 <-> M2 should NOT use Z direction";
    EXPECT_FALSE(mesh_graph.should_assign_z_direction(mesh_2, mesh_1))
        << "M2 <-> M1 should NOT use Z direction (bidirectional)";

    // Clean up
    std::filesystem::remove(test_file);
}

TEST(MeshGraphDescriptorTests, AssignZDirectionGraphTopologyInMeshGraph) {
    // Test that assign_z_direction flag from graph topology is properly tracked in MeshGraph
    const std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 2, 2 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 1 ] }
        }

        graph_descriptors: {
          name: "G0"
          type: "FABRIC"
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 1 } }
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 2 } }
          graph_topology: {
            layout_type: ALL_TO_ALL
            channels: { count: 2 }
            assign_z_direction: true
          }
        }

        top_level_instance: { graph: { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    // Create a temporary file for the test
    const std::filesystem::path test_file =
        std::filesystem::temp_directory_path() / "test_assign_z_direction_graph_topology.textproto";
    {
        std::ofstream file(test_file);
        file << text_proto;
    }

    // Cluster type doesn't matter
    const tt::tt_metal::ClusterType cluster_type = tt::tt_metal::ClusterType::BLACKHOLE_GALAXY;
    EXPECT_NO_THROW(tt::tt_fabric::MeshGraph mesh_graph(cluster_type, test_file.string()));

    tt::tt_fabric::MeshGraph mesh_graph(cluster_type, test_file.string());

    // Test should_assign_z_direction method for all mesh pairs
    tt::tt_fabric::MeshId mesh_0(0);
    tt::tt_fabric::MeshId mesh_1(1);
    tt::tt_fabric::MeshId mesh_2(2);

    // All pairs should use Z direction (ALL-to-ALL with assign_z_direction: true)
    EXPECT_TRUE(mesh_graph.should_assign_z_direction(mesh_0, mesh_1))
        << "M0 <-> M1 should use Z direction (ALL-to-ALL topology)";
    EXPECT_TRUE(mesh_graph.should_assign_z_direction(mesh_1, mesh_0))
        << "M1 <-> M0 should use Z direction (bidirectional)";

    EXPECT_TRUE(mesh_graph.should_assign_z_direction(mesh_0, mesh_2))
        << "M0 <-> M2 should use Z direction (ALL-to-ALL topology)";
    EXPECT_TRUE(mesh_graph.should_assign_z_direction(mesh_2, mesh_0))
        << "M2 <-> M0 should use Z direction (bidirectional)";

    EXPECT_TRUE(mesh_graph.should_assign_z_direction(mesh_1, mesh_2))
        << "M1 <-> M2 should use Z direction (ALL-to-ALL topology)";
    EXPECT_TRUE(mesh_graph.should_assign_z_direction(mesh_2, mesh_1))
        << "M2 <-> M1 should use Z direction (bidirectional)";

    // Clean up
    std::filesystem::remove(test_file);
}

TEST(MeshGraphDescriptorTests, PinningsParsing) {
    // Test that pinnings are parsed correctly from textproto
    const std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 8, 4 ] }
          channels: { count: 4 }
          host_topology: { dims: [ 1, 1 ] }
        }

        pinnings: {
          logical_fabric_node_id: { mesh_id: 0 chip_id: 0 }
          physical_asic_position: { tray_id: 1 asic_location: 1 }
        }

        pinnings: {
          logical_fabric_node_id: { mesh_id: 0 chip_id: 31 }
          physical_asic_position: { tray_id: 4 asic_location: 1 }
        }

        top_level_instance: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
    )proto";

    EXPECT_NO_THROW(MeshGraphDescriptor desc(text_proto));

    MeshGraphDescriptor desc(text_proto);

    // Check that pinnings were extracted as one group per entry (both on mesh 0)
    const auto& pinnings = desc.get_pinnings();
    ASSERT_EQ(pinnings.size(), 1u);
    ASSERT_EQ(pinnings.at(MeshId{0}).size(), 2u) << "Should have 2 pinning groups";

    // Check first pinning: (mesh 0, chip 0) -> (tray 1, location 1)
    const auto& pinning1 = pinnings.at(MeshId{0})[0];
    ASSERT_EQ(pinning1.fabric_nodes.size(), 1) << "First pinning should have 1 fabric node";
    ASSERT_EQ(pinning1.asic_positions.size(), 1) << "First pinning should have 1 ASIC position";
    EXPECT_EQ(*pinning1.asic_positions[0].first, 1) << "First pinning should have tray_id 1";
    EXPECT_EQ(*pinning1.asic_positions[0].second, 1) << "First pinning should have asic_location 1";
    EXPECT_EQ(*pinning1.fabric_nodes[0].mesh_id, 0) << "First pinning should have mesh_id 0";
    EXPECT_EQ(pinning1.fabric_nodes[0].chip_id, 0) << "First pinning should have chip_id 0";

    // Check second pinning: (mesh 0, chip 31) -> (tray 4, location 1)
    const auto& pinning2 = pinnings.at(MeshId{0})[1];
    ASSERT_EQ(pinning2.fabric_nodes.size(), 1) << "Second pinning should have 1 fabric node";
    ASSERT_EQ(pinning2.asic_positions.size(), 1) << "Second pinning should have 1 ASIC position";
    EXPECT_EQ(*pinning2.asic_positions[0].first, 4) << "Second pinning should have tray_id 4";
    EXPECT_EQ(*pinning2.asic_positions[0].second, 1) << "Second pinning should have asic_location 1";
    EXPECT_EQ(*pinning2.fabric_nodes[0].mesh_id, 0) << "Second pinning should have mesh_id 0";
    EXPECT_EQ(pinning2.fabric_nodes[0].chip_id, 31) << "Second pinning should have chip_id 31";
}

TEST(MeshGraphDescriptorTests, PinningsMeshIdRegexRangeExpandsPerMesh) {
    // mesh_id_regex "0-2" (range) should replicate the entry for meshes 0,1,2 -> one all-to-all group each.
    const std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 2, 2 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 1 ] }
        }
        graph_descriptors: {
          name: "G0"
          type: "FABRIC"
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 1 } }
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 2 } }
        }
        pinnings: {
          logical_fabric_node_id: { mesh_id_regex: "0-2" chip_id: 0 }
          logical_fabric_node_id: { mesh_id_regex: "0-2" chip_id: 3 }
          physical_asic_position: { tray_id: 1 asic_location: 1 }
          physical_asic_position: { tray_id: 2 asic_location: 1 }
        }
        top_level_instance: { graph: { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    MeshGraphDescriptor desc(text_proto);
    const auto& pinnings = desc.get_pinnings();
    ASSERT_EQ(pinnings.size(), 3u) << "One group per matched mesh (0,1,2)";
    for (uint32_t m = 0; m < 3; ++m) {
        const auto& g = pinnings.at(MeshId{m}).front();
        ASSERT_EQ(g.fabric_nodes.size(), 2u);
        EXPECT_EQ(*g.fabric_nodes[0].mesh_id, m);
        EXPECT_EQ(g.fabric_nodes[0].chip_id, 0u);
        EXPECT_EQ(*g.fabric_nodes[1].mesh_id, m);
        EXPECT_EQ(g.fabric_nodes[1].chip_id, 3u);
        ASSERT_EQ(g.asic_positions.size(), 2u);
    }
}

TEST(MeshGraphDescriptorTests, PinningsMeshIdRegexEvenOddParity) {
    // Regex (not a range) selecting even vs odd mesh ids, plus chip_id_regex as a range.
    const std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 2, 2 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 1 ] }
        }
        graph_descriptors: {
          name: "G0"
          type: "FABRIC"
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 1 } }
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 2 } }
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 3 } }
        }
        pinnings: {
          logical_fabric_node_id: { mesh_id_regex: "[0-9]*[02468]" chip_id_regex: "0-3" }
          physical_asic_position: { tray_id: 1 asic_location: 3 }
        }
        top_level_instance: { graph: { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    MeshGraphDescriptor desc(text_proto);
    const auto& pinnings = desc.get_pinnings();
    ASSERT_EQ(pinnings.size(), 2u) << "Even meshes 0 and 2";
    EXPECT_EQ(*pinnings.at(MeshId{0}).front().fabric_nodes.front().mesh_id, 0u);
    EXPECT_EQ(*pinnings.at(MeshId{2}).front().fabric_nodes.front().mesh_id, 2u);
    // chip_id_regex "0-3" over a 2x2 (4-chip) mesh -> chips 0,1,2,3.
    ASSERT_EQ(pinnings.at(MeshId{0}).front().fabric_nodes.size(), 4u);
    for (uint32_t c = 0; c < 4; ++c) {
        EXPECT_EQ(pinnings.at(MeshId{0}).front().fabric_nodes[c].chip_id, c);
    }
}

TEST(MeshGraphDescriptorTests, PinningsRegexMixedWithNonRegexError) {
    // Mixing regex and literal logical_fabric_node_id in one pinning entry is ambiguous and must fail.
    const std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 2, 2 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 1 ] }
        }
        graph_descriptors: {
          name: "G0"
          type: "FABRIC"
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 1 } }
        }
        pinnings: {
          logical_fabric_node_id: { mesh_id_regex: "0-1" chip_id: 0 }
          logical_fabric_node_id: { mesh_id: 0 chip_id: 3 }
          physical_asic_position: { tray_id: 1 asic_location: 1 }
        }
        top_level_instance: { graph: { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    EXPECT_THAT(
        [&]() { MeshGraphDescriptor desc(text_proto); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::AllOf(
            ::testing::HasSubstr("Failed to validate MeshGraphDescriptor textproto"),
            ::testing::HasSubstr("mixes regex and non-regex"))));
}

TEST(MeshGraphDescriptorTests, PinningsRegexMeshIdAndNumericMeshIdError) {
    const std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 2, 2 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 1 ] }
        }
        graph_descriptors: {
          name: "G0"
          type: "FABRIC"
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 1 } }
        }
        pinnings: {
          logical_fabric_node_id { mesh_id_regex: "0-1" mesh_id: 0 chip_id: 0 }
          physical_asic_position { tray_id: 1 asic_location: 1 }
        }
        top_level_instance: { graph: { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    EXPECT_THAT(
        [&]() { MeshGraphDescriptor desc(text_proto); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::AllOf(
            ::testing::HasSubstr("Failed to validate MeshGraphDescriptor textproto"),
            ::testing::HasSubstr("mesh_id_regex and mesh_id"))));
}

TEST(MeshGraphDescriptorTests, PinningsPhysicalRegexExpands) {
    const std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 2, 2 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 1 ] }
        }
        graph_descriptors: {
          name: "G0"
          type: "FABRIC"
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
        }
        pinnings: {
          logical_fabric_node_id: { mesh_id: 0 chip_id: 0 }
          physical_asic_position: { tray_id_regex: "1-4" asic_location: 3 }
        }
        pinnings: {
          logical_fabric_node_id: { mesh_id: 0 chip_id: 1 }
          physical_asic_position: { tray_id: 1 asic_location_regex: "2,3,6,7" }
        }
        top_level_instance: { graph: { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    MeshGraphDescriptor desc(text_proto);
    const auto& pinnings = desc.get_pinnings();
    ASSERT_EQ(pinnings.size(), 1u);
    const auto& groups = pinnings.at(MeshId{0});
    ASSERT_EQ(groups.size(), 2u);

    ASSERT_EQ(groups[0].asic_positions.size(), 4u);
    for (uint32_t tray : {1u, 2u, 3u, 4u}) {
        bool found = false;
        for (const auto& pos : groups[0].asic_positions) {
            if (*pos.first == tray && *pos.second == 3u) {
                found = true;
            }
        }
        EXPECT_TRUE(found) << "Expected tray " << tray << " asic_location 3";
    }

    ASSERT_EQ(groups[1].asic_positions.size(), 4u);
    for (uint32_t loc : {2u, 3u, 6u, 7u}) {
        bool found = false;
        for (const auto& pos : groups[1].asic_positions) {
            if (*pos.first == 1u && *pos.second == loc) {
                found = true;
            }
        }
        EXPECT_TRUE(found) << "Expected tray 1 asic_location " << loc;
    }
}

TEST(MeshGraphDescriptorTests, PinningsRepeatedNodeKeptAsSeparateGroups) {
    // The same fabric node may appear in several entries. Each entry is carried through as its own group;
    // consumers filter to the positions present on the mesh being solved, so whether the entries can be
    // satisfied together is decided there, not at parse time.
    const std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 2, 2 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 1 ] }
        }

        pinnings: {
          logical_fabric_node_id: { mesh_id: 0 chip_id: 0 }
          physical_asic_position: { tray_id: 1 asic_location: 1 }
        }

        pinnings: {
          logical_fabric_node_id: { mesh_id: 0 chip_id: 0 }
          physical_asic_position: { tray_id: 2 asic_location: 2 }
        }

        top_level_instance: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
    )proto";

    MeshGraphDescriptor desc(text_proto);
    const auto& pinnings = desc.get_pinnings();

    ASSERT_EQ(pinnings.size(), 1u);
    const auto& groups = pinnings.at(MeshId{0});
    ASSERT_EQ(groups.size(), 2u);
    const FabricNodeId node(MeshId{0}, 0);
    for (const auto& group : groups) {
        ASSERT_EQ(group.fabric_nodes.size(), 1u);
        EXPECT_EQ(group.fabric_nodes[0], node);
        ASSERT_EQ(group.asic_positions.size(), 1u);
    }
    EXPECT_EQ(*groups[0].asic_positions[0].first, 1u);
    EXPECT_EQ(*groups[0].asic_positions[0].second, 1u);
    EXPECT_EQ(*groups[1].asic_positions[0].first, 2u);
    EXPECT_EQ(*groups[1].asic_positions[0].second, 2u);
}

TEST(MeshGraphDescriptorTests, PinningsEmpty) {
    // Test that empty pinnings section is valid
    const std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 2, 2 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 1 ] }
        }

        top_level_instance: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
    )proto";

    EXPECT_NO_THROW(MeshGraphDescriptor desc(text_proto));

    MeshGraphDescriptor desc(text_proto);

    // Check that pinnings map is empty
    const auto& pinnings = desc.get_pinnings();
    EXPECT_TRUE(pinnings.empty()) << "Should have no pinnings when none are specified";
}

// Finding B -- a directional device-level (STRICT) inter-mesh connection authored high->low is rejected at parse
// time (tracked by https://github.com/tenstorrent/tt-metal/issues/50292).
//
// MeshGraphDescriptor only expands a connection into both directions when it is NOT directional, so a
// `directional: true` device-level connection authored M1.D0->M0.D0 would be stored ONLY in its authored
// direction requested_intermesh_ports[1][0]. In the control plane the peer (M0) side then never gathers the
// cable, the two-sided connection_hash join drops the link, and strict binding resolves 0 routers -> a confusing
// downstream hard-fatal. Until directionality is tracked as a first-class property (gather bidirectional but
// routing one-way), MeshGraphDescriptor hard-fails up front. This test pins that behavior for the device-level
// STRICT case; rework it into a real end-to-end (control-plane gather + budget + one-way routing) test once
// #50292 is implemented.
TEST(MeshGraphDescriptorTests, FindingB_DirectionalDeviceLevelStrictIntermeshIsRejected) {
    // Two 1x2 BH meshes with a single directional device-level STRICT connection authored HIGH->LOW
    // (M1.D0 -> M0.D0).
    const std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: BLACKHOLE
          device_topology: {
            dims: [ 1, 2 ]
            dim_types: [ LINE, LINE ]
          }
          channels: { count: 2 policy: RELAXED }
          host_topology: { dims: [ 1, 1 ] }
        }

        graph_descriptors: {
          name: "G0"
          type: "FABRIC"
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 1 } }
          connections: {
            nodes: { mesh: { mesh_descriptor: "M0" mesh_id: 1 device_id: 0 } }
            nodes: { mesh: { mesh_descriptor: "M0" mesh_id: 0 device_id: 0 } }
            channels: { count: 2 policy: STRICT }
            directional: true
          }
        }

        top_level_instance: { graph: { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    EXPECT_THAT(
        ([&]() { MeshGraphDescriptor desc(text_proto); }),
        ::testing::ThrowsMessage<std::runtime_error>(::testing::AllOf(
            ::testing::HasSubstr("directional inter-mesh connection"),
            ::testing::HasSubstr("not fully supported"),
            ::testing::HasSubstr("50292"))));
}

TEST(MeshGraphDescriptorTests, PinningsAllToAll) {
    // A single pinning entry listing 2 logical nodes and 2 physical positions is stored as one
    // many-to-many group (any listed node may map to any listed position).
    const std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 2, 2 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 1 ] }
        }

        pinnings: {
          logical_fabric_node_id: { mesh_id: 0 chip_id: 0 }
          logical_fabric_node_id: { mesh_id: 0 chip_id: 3 }
          physical_asic_position: { tray_id: 1 asic_location: 1 }
          physical_asic_position: { tray_id: 4 asic_location: 1 }
        }

        top_level_instance: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
    )proto";

    MeshGraphDescriptor desc(text_proto);

    const auto& pinnings = desc.get_pinnings();
    ASSERT_EQ(pinnings.size(), 1) << "The entry should be stored as a single many-to-many group";

    const auto& group = pinnings.at(MeshId{0}).front();
    ASSERT_EQ(group.fabric_nodes.size(), 2) << "Group should list 2 fabric nodes";
    ASSERT_EQ(group.asic_positions.size(), 2) << "Group should list 2 ASIC positions";

    auto has_node = [&](uint32_t chip_id) {
        for (const auto& node : group.fabric_nodes) {
            if (*node.mesh_id == 0 && node.chip_id == chip_id) {
                return true;
            }
        }
        return false;
    };
    auto has_position = [&](uint32_t tray_id, uint32_t asic_location) {
        for (const auto& pos : group.asic_positions) {
            if (*pos.first == tray_id && *pos.second == asic_location) {
                return true;
            }
        }
        return false;
    };

    EXPECT_TRUE(has_node(0));
    EXPECT_TRUE(has_node(3));
    EXPECT_TRUE(has_position(1, 1));
    EXPECT_TRUE(has_position(4, 1));
}

TEST(MeshGraphDescriptorTests, PinningsRegexMalformedRangeError) {
    const std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 2, 2 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 1 ] }
        }
        graph_descriptors: {
          name: "G0"
          type: "FABRIC"
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
        }
        pinnings: {
          logical_fabric_node_id: { mesh_id_regex: "1-" chip_id: 0 }
          physical_asic_position: { tray_id: 1 asic_location: 1 }
        }
        top_level_instance: { graph: { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    EXPECT_THAT(
        [&]() { MeshGraphDescriptor desc(text_proto); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::AllOf(
            ::testing::HasSubstr("Failed to validate MeshGraphDescriptor textproto"),
            ::testing::HasSubstr("malformed range token"))));
}

TEST(MeshGraphDescriptorTests, PinningsRegexInvalidRegexError) {
    const std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 2, 2 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 1 ] }
        }
        graph_descriptors: {
          name: "G0"
          type: "FABRIC"
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
        }
        pinnings: {
          logical_fabric_node_id: { mesh_id_regex: "[unclosed" chip_id: 0 }
          physical_asic_position: { tray_id: 1 asic_location: 1 }
        }
        top_level_instance: { graph: { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    EXPECT_THAT(
        [&]() { MeshGraphDescriptor desc(text_proto); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::AllOf(
            ::testing::HasSubstr("Failed to validate MeshGraphDescriptor textproto"),
            ::testing::HasSubstr("invalid regex"))));
}

TEST(MeshGraphDescriptorTests, PinningsRegexOverlappingMeshIdsExpandPerMesh) {
    // mesh_id_regex "0-2" and "2-3" both match mesh 2, so mesh 2's chip 0 lands in a group from each entry.
    // Overlap is legal: expansion yields one group per matched mesh per entry.
    const std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 2, 2 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 1 ] }
        }
        graph_descriptors: {
          name: "G0"
          type: "FABRIC"
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 1 } }
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 2 } }
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 3 } }
        }
        pinnings: {
          logical_fabric_node_id: { mesh_id_regex: "0-2" chip_id: 0 }
          physical_asic_position: { tray_id: 1 asic_location: 1 }
        }
        pinnings: {
          logical_fabric_node_id: { mesh_id_regex: "2-3" chip_id: 0 }
          physical_asic_position: { tray_id: 2 asic_location: 1 }
        }
        top_level_instance: { graph: { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    MeshGraphDescriptor desc(text_proto);
    const auto& pinnings = desc.get_pinnings();

    // Meshes 0, 1, 2 from the first entry plus meshes 2, 3 from the second.
    ASSERT_EQ(pinnings.size(), 4u);
    std::map<uint32_t, uint32_t> groups_per_mesh;
    for (const auto& [_, groups] : pinnings) {
        for (const auto& group : groups) {
            ASSERT_EQ(group.fabric_nodes.size(), 1u);
            EXPECT_EQ(group.fabric_nodes[0].chip_id, 0u);
            groups_per_mesh[*group.fabric_nodes[0].mesh_id]++;
        }
    }
    EXPECT_EQ(groups_per_mesh[0], 1u);
    EXPECT_EQ(groups_per_mesh[1], 1u);
    EXPECT_EQ(groups_per_mesh[2], 2u);
    EXPECT_EQ(groups_per_mesh[3], 1u);
}

TEST(MeshGraphDescriptorTests, PinningsLiteralMultiMeshSplitsByMesh) {
    // A non-regex entry that lists nodes from two meshes must emit one group per mesh, matching the
    // regex path, so get_pinnings() can look up without filtering mixed-mesh groups.
    const std::string text_proto = R"proto(
        mesh_descriptors: {
          name: "M0"
          arch: WORMHOLE_B0
          device_topology: { dims: [ 2, 2 ] }
          channels: { count: 1 }
          host_topology: { dims: [ 1, 1 ] }
        }
        graph_descriptors: {
          name: "G0"
          type: "FABRIC"
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 0 } }
          instances: { mesh: { mesh_descriptor: "M0" mesh_id: 1 } }
        }
        pinnings: {
          logical_fabric_node_id: { mesh_id: 0 chip_id: 0 }
          logical_fabric_node_id: { mesh_id: 0 chip_id: 3 }
          logical_fabric_node_id: { mesh_id: 1 chip_id: 0 }
          physical_asic_position: { tray_id: 1 asic_location: 3 }
        }
        top_level_instance: { graph: { graph_descriptor: "G0" graph_id: 0 } }
    )proto";

    MeshGraphDescriptor desc(text_proto);
    const auto& pinnings = desc.get_pinnings();
    ASSERT_EQ(pinnings.size(), 2u);

    const auto& mesh0 = pinnings.at(MeshId{0});
    ASSERT_EQ(mesh0.size(), 1u);
    ASSERT_EQ(mesh0[0].fabric_nodes.size(), 2u);
    EXPECT_EQ(*mesh0[0].fabric_nodes[0].mesh_id, 0u);
    EXPECT_EQ(mesh0[0].fabric_nodes[0].chip_id, 0u);
    EXPECT_EQ(*mesh0[0].fabric_nodes[1].mesh_id, 0u);
    EXPECT_EQ(mesh0[0].fabric_nodes[1].chip_id, 3u);

    const auto& mesh1 = pinnings.at(MeshId{1});
    ASSERT_EQ(mesh1.size(), 1u);
    ASSERT_EQ(mesh1[0].fabric_nodes.size(), 1u);
    EXPECT_EQ(*mesh1[0].fabric_nodes[0].mesh_id, 1u);
    EXPECT_EQ(mesh1[0].fabric_nodes[0].chip_id, 0u);

    EXPECT_FALSE(pinnings.contains(MeshId{2}));
}

TEST(MeshGraphDescriptorTests, VectorReallocPreservesConnectionsByTypeLookup) {
    const char* tt_metal_home = std::getenv("TT_METAL_HOME");
    ASSERT_NE(tt_metal_home, nullptr) << "TT_METAL_HOME environment variable must be set";
    const std::filesystem::path path_4x4 =
        std::filesystem::path(tt_metal_home) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/bh_galaxy_single_4x4_mesh.textproto";
    const std::filesystem::path path_dual =
        std::filesystem::path(tt_metal_home) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/bh_galaxy_dual_2x4_intermesh.textproto";
    ASSERT_TRUE(std::filesystem::exists(path_4x4));
    ASSERT_TRUE(std::filesystem::exists(path_dual));

    std::vector<MeshGraphDescriptor> mgds;
    // No reserve(): second emplace typically reallocates and move-constructs elements — connection-by-type keys must
    // remain valid (owning std::string keys, not string_views into moved-from InstanceData).
    mgds.emplace_back(path_4x4);
    const size_t mesh_conn_count = mgds[0].connections_by_type("MESH").size();
    EXPECT_GT(mesh_conn_count, 0u) << "4x4 grid MGD should synthesize intra-mesh MESH connections";

    mgds.emplace_back(path_dual);

    EXPECT_EQ(mgds[0].connections_by_type("MESH").size(), mesh_conn_count)
        << "First MGD's MESH connection index must survive vector reallocation";
    EXPECT_FALSE(mgds[1].connections_by_type("FABRIC").empty())
        << "Dual MGD should retain FABRIC connections after emplace";
// Verify the skip_links pattern (axis=ROW start=2 step=4) expands into the expected intra-mesh Z edges
// on the shared 8x4 [LINE, RING] descriptor (also used by the routing/lowering tests).
// skip_links expands into the expected intra-mesh Z edges on the 8x4 [LINE, RING] descriptor.
TEST(MeshGraphDescriptorTests, SkipLinks8x4) {
// express_links expands into the expected intra-mesh Z edges on the 8x4 [LINE, RING] descriptor.
TEST(MeshGraphDescriptorTests, ExpressLinks8x4) {
    const std::filesystem::path desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/express_links_8x4_mesh_graph_descriptor.textproto";

    EXPECT_NO_THROW(
        tt::tt_fabric::MeshGraph mesh_graph(tt::tt_metal::ClusterType::BLACKHOLE_GALAXY, desc_path.string()));

    tt::tt_fabric::MeshGraph mesh_graph(tt::tt_metal::ClusterType::BLACKHOLE_GALAXY, desc_path.string());
    const auto& intra = mesh_graph.get_intra_mesh_connectivity();
    ASSERT_EQ(intra.size(), 1u);
    const auto& m0 = intra[0];
    ASSERT_EQ(m0.size(), 32u);  // 8x4 = 32 chips

    // wrap: LINE on the step-4 pattern keeps only block [2,5]; the wrapping block is dropped even
    // though the row axis is RING. Row 0 <-> row 7 is the ordinary wrap, not an express link. chip = row*4 + col.
    const std::vector<std::pair<int, int>> expected_express_edges = {{8, 20}, {9, 21}, {10, 22}, {11, 23}};

    for (const auto& [a, b] : expected_express_edges) {
        EXPECT_EQ(m0[a].count(b), 1u) << "missing express edge " << a << " -> " << b;
        EXPECT_EQ(m0[b].count(a), 1u) << "missing reverse express edge " << b << " -> " << a;
        if (m0[a].count(b) && m0[b].count(a)) {
            EXPECT_EQ(m0[a].at(b).port_direction, tt::tt_fabric::RoutingDirection::Z);
            EXPECT_EQ(m0[b].at(a).port_direction, tt::tt_fabric::RoutingDirection::Z);
        }
    }

    EXPECT_EQ(m0[4].count(24), 0u);  // row 1 is not a block endpoint, so it gets no express link

    // chip 8 keeps its 4 base-grid neighbors plus the one express edge
    EXPECT_EQ(m0[8].count(4), 1u);
    EXPECT_EQ(m0[8].count(12), 1u);
    EXPECT_EQ(m0[8].count(9), 1u);
    EXPECT_EQ(m0[8].count(11), 1u);
    EXPECT_EQ(m0[8].size(), 5u);

    // 4 bidirectional express edges = 8 directed Z entries, no others
    int z_directed = 0;
    for (int c = 0; c < 32; ++c) {
        for (const auto& [nb, edge] : m0[c]) {
            if (edge.port_direction == tt::tt_fabric::RoutingDirection::Z) {
                ++z_directed;
            }
        }
    }
    EXPECT_EQ(z_directed, 8) << "expected exactly 4 bidirectional express edges (8 directed Z entries)";
}

// express_links (two ROW patterns) expand into 48 Z edges on the 32x4 [RING, RING] descriptor.
TEST(MeshGraphDescriptorTests, ExpressLinks32x4) {
    const std::filesystem::path desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/express_links_32x4_mesh_graph_descriptor.textproto";

    EXPECT_NO_THROW(
        tt::tt_fabric::MeshGraph mesh_graph(tt::tt_metal::ClusterType::BLACKHOLE_GALAXY, desc_path.string()));

    tt::tt_fabric::MeshGraph mesh_graph(tt::tt_metal::ClusterType::BLACKHOLE_GALAXY, desc_path.string());
    const auto& intra = mesh_graph.get_intra_mesh_connectivity();
    ASSERT_EQ(intra.size(), 1u);
    const auto& m0 = intra[0];
    ASSERT_EQ(m0.size(), 128u);  // 32x4 = 128 chips

    // dim 0 (32 rows, RING). chip = row*4 + col. Two patterns:
    //   start=2 step=4 -> 8 row pairs (last wraps)
    //   start=0 step=8 -> 4 row pairs
    const std::vector<std::pair<int, int>> row_blocks = {
        {2, 5},
        {6, 9},
        {10, 13},
        {14, 17},
        {18, 21},
        {22, 25},
        {26, 29},
        {30, 1},  // start=2 step=4
        {0, 7},
        {8, 15},
        {16, 23},
        {24, 31}};  // start=0 step=8
    for (const auto& [ra, rb] : row_blocks) {
        for (int col = 0; col < 4; ++col) {
            const int a = ra * 4 + col;
            const int b = rb * 4 + col;
            EXPECT_EQ(m0[a].count(b), 1u) << "missing express edge " << a << " -> " << b;
            EXPECT_EQ(m0[b].count(a), 1u) << "missing reverse express edge " << b << " -> " << a;
            if (m0[a].count(b) && m0[b].count(a)) {
                EXPECT_EQ(m0[a].at(b).port_direction, tt::tt_fabric::RoutingDirection::Z);
                EXPECT_EQ(m0[b].at(a).port_direction, tt::tt_fabric::RoutingDirection::Z);
            }
        }
    }

    // (8 + 4) blocks x 4 columns = 48 bidirectional express edges = 96 directed Z entries, no others
    int z_directed = 0;
    for (int c = 0; c < 128; ++c) {
        for (const auto& [nb, edge] : m0[c]) {
            if (edge.port_direction == tt::tt_fabric::RoutingDirection::Z) {
                ++z_directed;
            }
        }
    }
    EXPECT_EQ(z_directed, 96) << "expected exactly 48 bidirectional express edges (96 directed Z entries)";
}

}  // namespace tt::tt_fabric::fabric_router_tests
