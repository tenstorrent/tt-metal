// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "tt_metal/fabric/builder/connection_registry.hpp"
#include "tt_metal/fabric/builder/router_connection_mapping.hpp"
#include "tt_metal/fabric/fabric_router_channel_mapping.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

using namespace tt::tt_fabric;

/**
 * FabricBuilder Local Connection Tests
 *
 * These tests validate FabricBuilder's establishment of local mesh↔Z connections:
 * 1. Z router detection on a device
 * 2. Local mesh↔Z connection establishment
 * 3. Variable mesh router count handling (2-4 routers)
 * 4. Connection registry validation for full device scenarios
 *
 * Note: These are conceptual tests simulating FabricBuilder connection logic
 * without requiring actual device/UMD initialization.
 *
 * Test Coverage Summary:
 * ┌──────────────────────────────────────────────────────────────────────────────────────┐
 * │ Category                    │ Test Name                                │ Focus        │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Z Router Detection          │ DetectZRouter_FullDevice                 │ Has Z        │
 * │                             │ DetectZRouter_NoZRouter                  │ No Z         │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Full Device (4 Mesh + Z)    │ FullDevice_4Mesh1Z_AllConnections        │ All conns    │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Edge Device (2-3 Mesh + Z)  │ EdgeDevice_2Mesh1Z_Connections           │ 2 routers    │
 * │                             │ EdgeDevice_3Mesh1Z_Connections           │ 3 routers    │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ VC Assignment (Positive)    │ VCAssignment_MeshToZ_VC0                 │ Mesh→Z VC0   │
 * │                             │ VCAssignment_ZToMesh_VC1                 │ Z→Mesh VC1   │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ VC Assignment (Negative)    │ VCAssignment_MeshToZ_VC1_Negative        │ No VC1       │
 * │                             │ VCAssignment_ZToMesh_VC0_Negative        │ No VC0       │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Connection Validation       │ ZRouter_ConnectsToCorrectDirections      │ Directions   │
 * │                             │ NoDuplicateConnections                   │ No dupes     │
 * │                             │ ConnectionOrder_ZFirst                   │ Order indep  │
 * │                             │ ConnectionOrder_MeshFirst                │ Order indep  │
 * └─────────────────────────────┴──────────────────────────────────────────┴──────────────┘
 *
 * Total: 13 tests across 6 categories
 */

class FabricBuilderLocalConnectionsTest : public ::testing::Test {
protected:
    void SetUp() override {
        registry_ = std::make_shared<ConnectionRegistry>();
    }

    // Helper to simulate router creation
    struct MockRouter {
        FabricNodeId node_id;
        RoutingDirection direction;
        FabricRouterChannelMapping channel_mapping;
        RouterConnectionMapping connection_mapping;
    };

    MockRouter create_mock_mesh_router(RoutingDirection dir, uint32_t router_id, bool has_z) {
        return MockRouter{
            .node_id = FabricNodeId(MeshId{0}, router_id),
            .direction = dir,
            .channel_mapping = FabricRouterChannelMapping(
                Topology::Mesh,
                false,  // No tensix
                RouterVariant::MESH,
                nullptr),  // No intermesh config for mock
            .connection_mapping = RouterConnectionMapping::for_mesh_router(
                Topology::Mesh,
                dir,
                has_z)
        };
    }

    MockRouter create_mock_z_router(uint32_t router_id) {
        static auto intermesh_config = IntermeshVCConfig::full_mesh();
        return MockRouter{
            .node_id = FabricNodeId(MeshId{0}, router_id),
            .direction = RoutingDirection::Z,
            .channel_mapping = FabricRouterChannelMapping(
                Topology::Mesh,
                false,
                RouterVariant::Z_ROUTER,
                &intermesh_config),  // Z routers require intermesh config
            .connection_mapping = RouterConnectionMapping::for_z_router()
        };
    }

    // Helper to simulate local connection establishment
    void establish_local_connections(MockRouter& source, const std::map<RoutingDirection, MockRouter*>& targets) {
        // Iterate through all sender keys in connection mapping
        auto all_receiver_keys = source.connection_mapping.get_all_receiver_keys();

        for (const auto& key : all_receiver_keys) {
            auto conn_targets = source.connection_mapping.get_downstream_targets(key.vc, key.receiver_channel);

            for (const auto& target : conn_targets) {
                if (target.type == ConnectionType::MESH_TO_Z || target.type == ConnectionType::Z_TO_MESH) {
                    TT_FATAL(
                        target.target_direction.has_value(),
                        "target_direction must have a value for MESH_TO_Z or Z_TO_MESH connections");
                    auto target_dir = target.target_direction.value();

                    if (!targets.contains(target_dir)) {
                        continue;  // Target doesn't exist (edge device)
                    }

                    auto* dest_router = targets.at(target_dir);

                    // Record connection
                    RouterConnectionRecord record{
                        .source_node = source.node_id,
                        .source_direction = source.direction,
                        .source_eth_chan = 0,
                        .source_vc = key.vc,
                        .source_receiver_channel = key.receiver_channel,
                        .dest_node = dest_router->node_id,
                        .dest_direction = dest_router->direction,
                        .dest_eth_chan = 0,
                        .dest_vc = target.target_vc,
                        .dest_sender_channel = 0,
                        .connection_type = target.type
                    };

                    registry_->record_connection(record);
                }
            }
        }
    }

    std::shared_ptr<ConnectionRegistry> registry_;
};

// ============================================================================
// Test 1: Z Router Detection
// ============================================================================

TEST_F(FabricBuilderLocalConnectionsTest, DetectZRouter_FullDevice) {
    // Simulate device with 4 mesh routers + 1 Z router
    std::vector<MockRouter> routers;

    routers.push_back(create_mock_mesh_router(RoutingDirection::N, 0, true));
    routers.push_back(create_mock_mesh_router(RoutingDirection::E, 1, true));
    routers.push_back(create_mock_mesh_router(RoutingDirection::S, 2, true));
    routers.push_back(create_mock_mesh_router(RoutingDirection::W, 3, true));
    routers.push_back(create_mock_z_router(100));

    // Count Z routers
    int z_count = 0;
    for (const auto& router : routers) {
        if (router.direction == RoutingDirection::Z) {
            z_count++;
        }
    }

    EXPECT_EQ(z_count, 1);
}

TEST_F(FabricBuilderLocalConnectionsTest, DetectZRouter_NoZRouter) {
    // Simulate device with only mesh routers
    std::vector<MockRouter> routers;

    routers.push_back(create_mock_mesh_router(RoutingDirection::N, 0, false));
    routers.push_back(create_mock_mesh_router(RoutingDirection::E, 1, false));
    routers.push_back(create_mock_mesh_router(RoutingDirection::S, 2, false));
    routers.push_back(create_mock_mesh_router(RoutingDirection::W, 3, false));

    // Count Z routers
    int z_count = 0;
    for (const auto& router : routers) {
        if (router.direction == RoutingDirection::Z) {
            z_count++;
        }
    }

    EXPECT_EQ(z_count, 0);
}

// ============================================================================
// Test 2: Full Device Connection Establishment (4 Mesh + 1 Z)
// ============================================================================

TEST_F(FabricBuilderLocalConnectionsTest, FullDevice_4Mesh1Z_AllConnections) {
    // Create routers
    auto mesh_n = create_mock_mesh_router(RoutingDirection::N, 0, true);
    auto mesh_e = create_mock_mesh_router(RoutingDirection::E, 1, true);
    auto mesh_s = create_mock_mesh_router(RoutingDirection::S, 2, true);
    auto mesh_w = create_mock_mesh_router(RoutingDirection::W, 3, true);
    auto z_router = create_mock_z_router(100);

    // Build local router maps
    std::map<RoutingDirection, MockRouter*> mesh_routers = {
        {RoutingDirection::N, &mesh_n},
        {RoutingDirection::E, &mesh_e},
        {RoutingDirection::S, &mesh_s},
        {RoutingDirection::W, &mesh_w}
    };

    std::map<RoutingDirection, MockRouter*> z_map = {
        {RoutingDirection::Z, &z_router}
    };

    // Establish Z→mesh connections
    establish_local_connections(z_router, mesh_routers);

    // Establish mesh→Z connections
    for (auto& [dir, mesh_router] : mesh_routers) {
        establish_local_connections(*mesh_router, z_map);
    }

    // Verify total connections: 4 Z_TO_MESH + 4 MESH_TO_Z = 8
    EXPECT_EQ(registry_->size(), 8);

    auto z_to_mesh = registry_->get_connections_by_type(ConnectionType::Z_TO_MESH);
    auto mesh_to_z = registry_->get_connections_by_type(ConnectionType::MESH_TO_Z);

    EXPECT_EQ(z_to_mesh.size(), 4);
    EXPECT_EQ(mesh_to_z.size(), 4);

    // Verify Z router connects to all 4 mesh routers
    auto z_outgoing = registry_->get_connections_by_source_node(z_router.node_id);
    EXPECT_EQ(z_outgoing.size(), 4);

    // Verify all mesh routers connect to Z
    for (const auto& [dir, mesh_router] : mesh_routers) {
        auto mesh_outgoing = registry_->get_connections_by_source_node(mesh_router->node_id);

        // Should have exactly 1 MESH_TO_Z connection
        int mesh_to_z_count = 0;
        for (const auto& conn : mesh_outgoing) {
            if (conn.connection_type == ConnectionType::MESH_TO_Z) {
                mesh_to_z_count++;
                EXPECT_EQ(conn.dest_node, z_router.node_id);
            }
        }
        EXPECT_EQ(mesh_to_z_count, 1);
    }
}

// ============================================================================
// Test 3: Edge Device Scenarios (2-3 Mesh Routers)
// ============================================================================

TEST_F(FabricBuilderLocalConnectionsTest, EdgeDevice_2Mesh1Z_Connections) {
    // Edge device with only N and E mesh routers
    auto mesh_n = create_mock_mesh_router(RoutingDirection::N, 0, true);
    auto mesh_e = create_mock_mesh_router(RoutingDirection::E, 1, true);
    auto z_router = create_mock_z_router(100);

    std::map<RoutingDirection, MockRouter*> mesh_routers = {
        {RoutingDirection::N, &mesh_n},
        {RoutingDirection::E, &mesh_e}
    };

    std::map<RoutingDirection, MockRouter*> z_map = {
        {RoutingDirection::Z, &z_router}
    };

    // Establish connections
    establish_local_connections(z_router, mesh_routers);

    for (auto& [dir, mesh_router] : mesh_routers) {
        establish_local_connections(*mesh_router, z_map);
    }

    // Verify: 2 Z_TO_MESH + 2 MESH_TO_Z = 4 total
    EXPECT_EQ(registry_->size(), 4);

    auto z_to_mesh = registry_->get_connections_by_type(ConnectionType::Z_TO_MESH);
    auto mesh_to_z = registry_->get_connections_by_type(ConnectionType::MESH_TO_Z);

    EXPECT_EQ(z_to_mesh.size(), 2);  // Only N and E
    EXPECT_EQ(mesh_to_z.size(), 2);

    // Verify Z router only connects to existing mesh routers
    auto z_outgoing = registry_->get_connections_by_source_node(z_router.node_id);
    EXPECT_EQ(z_outgoing.size(), 2);

    // Check that S and W directions were skipped
    std::set<RoutingDirection> connected_dirs;
    for (const auto& conn : z_outgoing) {
        connected_dirs.insert(conn.dest_direction);
    }

    EXPECT_TRUE(connected_dirs.contains(RoutingDirection::N));
    EXPECT_TRUE(connected_dirs.contains(RoutingDirection::E));
    EXPECT_FALSE(connected_dirs.contains(RoutingDirection::S));
    EXPECT_FALSE(connected_dirs.contains(RoutingDirection::W));
}

TEST_F(FabricBuilderLocalConnectionsTest, EdgeDevice_3Mesh1Z_Connections) {
    // Edge device with N, E, and S mesh routers
    auto mesh_n = create_mock_mesh_router(RoutingDirection::N, 0, true);
    auto mesh_e = create_mock_mesh_router(RoutingDirection::E, 1, true);
    auto mesh_s = create_mock_mesh_router(RoutingDirection::S, 2, true);
    auto z_router = create_mock_z_router(100);

    std::map<RoutingDirection, MockRouter*> mesh_routers = {
        {RoutingDirection::N, &mesh_n},
        {RoutingDirection::E, &mesh_e},
        {RoutingDirection::S, &mesh_s}
    };

    std::map<RoutingDirection, MockRouter*> z_map = {
        {RoutingDirection::Z, &z_router}
    };

    // Establish connections
    establish_local_connections(z_router, mesh_routers);

    for (auto& [dir, mesh_router] : mesh_routers) {
        establish_local_connections(*mesh_router, z_map);
    }

    // Verify: 3 Z_TO_MESH + 3 MESH_TO_Z = 6 total
    EXPECT_EQ(registry_->size(), 6);

    auto z_to_mesh = registry_->get_connections_by_type(ConnectionType::Z_TO_MESH);
    auto mesh_to_z = registry_->get_connections_by_type(ConnectionType::MESH_TO_Z);

    EXPECT_EQ(z_to_mesh.size(), 3);
    EXPECT_EQ(mesh_to_z.size(), 3);
}

// ============================================================================
// Test 4: VC Assignment Validation
// ============================================================================

TEST_F(FabricBuilderLocalConnectionsTest, VCAssignment_MeshToZ_VC0) {
    auto mesh_n = create_mock_mesh_router(RoutingDirection::N, 0, true);
    auto z_router = create_mock_z_router(100);

    std::map<RoutingDirection, MockRouter*> z_map = {
        {RoutingDirection::Z, &z_router}
    };

    establish_local_connections(mesh_n, z_map);

    auto connections = registry_->get_all_connections();
    ASSERT_EQ(connections.size(), 1);

    // Mesh VC0 → Z VC0
    EXPECT_EQ(connections[0].source_vc, 0);
    EXPECT_EQ(connections[0].dest_vc, 0);
    EXPECT_EQ(connections[0].connection_type, ConnectionType::MESH_TO_Z);
}

TEST_F(FabricBuilderLocalConnectionsTest, VCAssignment_ZToMesh_VC1) {
    auto mesh_n = create_mock_mesh_router(RoutingDirection::N, 0, true);
    auto z_router = create_mock_z_router(100);

    std::map<RoutingDirection, MockRouter*> mesh_routers = {
        {RoutingDirection::N, &mesh_n}
    };

    establish_local_connections(z_router, mesh_routers);

    auto connections = registry_->get_all_connections();
    ASSERT_EQ(connections.size(), 1);

    // Z VC1 → Mesh VC1
    EXPECT_EQ(connections[0].source_vc, 1);
    EXPECT_EQ(connections[0].dest_vc, 1);
    EXPECT_EQ(connections[0].connection_type, ConnectionType::Z_TO_MESH);
}

// Negative test: MESH_TO_Z should never use VC1
TEST_F(FabricBuilderLocalConnectionsTest, VCAssignment_MeshToZ_VC1_Negative) {
    auto mesh_n = create_mock_mesh_router(RoutingDirection::N, 0, true);
    auto z_router = create_mock_z_router(100);

    std::map<RoutingDirection, MockRouter*> z_map = {
        {RoutingDirection::Z, &z_router}
    };

    establish_local_connections(mesh_n, z_map);

    auto connections = registry_->get_all_connections();

    // Verify no MESH_TO_Z connections use VC1
    for (const auto& conn : connections) {
        if (conn.connection_type == ConnectionType::MESH_TO_Z) {
            EXPECT_NE(conn.source_vc, 1) << "MESH_TO_Z should never use VC1";
            EXPECT_NE(conn.dest_vc, 1) << "MESH_TO_Z should never target VC1";
        }
    }
}

// Negative test: Z_TO_MESH should never use VC0
TEST_F(FabricBuilderLocalConnectionsTest, VCAssignment_ZToMesh_VC0_Negative) {
    auto mesh_n = create_mock_mesh_router(RoutingDirection::N, 0, true);
    auto z_router = create_mock_z_router(100);

    std::map<RoutingDirection, MockRouter*> mesh_routers = {
        {RoutingDirection::N, &mesh_n}
    };

    establish_local_connections(z_router, mesh_routers);

    auto connections = registry_->get_all_connections();

    // Verify no Z_TO_MESH connections use VC0
    for (const auto& conn : connections) {
        if (conn.connection_type == ConnectionType::Z_TO_MESH) {
            EXPECT_NE(conn.source_vc, 0) << "Z_TO_MESH should never use VC0";
            EXPECT_NE(conn.dest_vc, 0) << "Z_TO_MESH should never target VC0";
        }
    }
}

// ============================================================================
// Test 5: Connection Direction Validation
// ============================================================================

TEST_F(FabricBuilderLocalConnectionsTest, ZRouter_ConnectsToCorrectDirections) {
    auto mesh_n = create_mock_mesh_router(RoutingDirection::N, 0, true);
    auto mesh_e = create_mock_mesh_router(RoutingDirection::E, 1, true);
    auto mesh_s = create_mock_mesh_router(RoutingDirection::S, 2, true);
    auto mesh_w = create_mock_mesh_router(RoutingDirection::W, 3, true);
    auto z_router = create_mock_z_router(100);

    std::map<RoutingDirection, MockRouter*> mesh_routers = {
        {RoutingDirection::N, &mesh_n},
        {RoutingDirection::E, &mesh_e},
        {RoutingDirection::S, &mesh_s},
        {RoutingDirection::W, &mesh_w}
    };

    establish_local_connections(z_router, mesh_routers);

    auto z_outgoing = registry_->get_connections_by_source_node(z_router.node_id);
    ASSERT_EQ(z_outgoing.size(), 4);

    // Verify each direction is connected exactly once
    std::map<RoutingDirection, int> direction_counts;
    for (const auto& conn : z_outgoing) {
        direction_counts[conn.dest_direction]++;
    }

    EXPECT_EQ(direction_counts[RoutingDirection::N], 1);
    EXPECT_EQ(direction_counts[RoutingDirection::E], 1);
    EXPECT_EQ(direction_counts[RoutingDirection::S], 1);
    EXPECT_EQ(direction_counts[RoutingDirection::W], 1);
}

// ============================================================================
// Test 6: No Duplicate Connections
// ============================================================================

TEST_F(FabricBuilderLocalConnectionsTest, NoDuplicateConnections) {
    auto mesh_n = create_mock_mesh_router(RoutingDirection::N, 0, true);
    auto z_router = create_mock_z_router(100);

    std::map<RoutingDirection, MockRouter*> mesh_routers = {
        {RoutingDirection::N, &mesh_n}
    };

    std::map<RoutingDirection, MockRouter*> z_map = {
        {RoutingDirection::Z, &z_router}
    };

    // Establish connections
    establish_local_connections(z_router, mesh_routers);
    establish_local_connections(mesh_n, z_map);

    // Should have exactly 2 connections (1 each direction)
    EXPECT_EQ(registry_->size(), 2);

    // Verify no duplicates by checking unique (source, dest, vc) tuples
    std::set<std::tuple<FabricNodeId, FabricNodeId, uint32_t>> unique_conns;
    for (const auto& conn : registry_->get_all_connections()) {
        unique_conns.insert({conn.source_node, conn.dest_node, conn.source_vc});
    }

    EXPECT_EQ(unique_conns.size(), 2);
}

// ============================================================================
// Test 7: Connection Order Independence
// ============================================================================

TEST_F(FabricBuilderLocalConnectionsTest, ConnectionOrder_ZFirst) {
    auto mesh_n = create_mock_mesh_router(RoutingDirection::N, 0, true);
    auto z_router = create_mock_z_router(100);

    std::map<RoutingDirection, MockRouter*> mesh_routers = {
        {RoutingDirection::N, &mesh_n}
    };

    std::map<RoutingDirection, MockRouter*> z_map = {
        {RoutingDirection::Z, &z_router}
    };

    // Z first, then mesh
    establish_local_connections(z_router, mesh_routers);
    establish_local_connections(mesh_n, z_map);

    EXPECT_EQ(registry_->size(), 2);

    auto z_to_mesh = registry_->get_connections_by_type(ConnectionType::Z_TO_MESH);
    auto mesh_to_z = registry_->get_connections_by_type(ConnectionType::MESH_TO_Z);

    EXPECT_EQ(z_to_mesh.size(), 1);
    EXPECT_EQ(mesh_to_z.size(), 1);
}

TEST_F(FabricBuilderLocalConnectionsTest, ConnectionOrder_MeshFirst) {
    auto mesh_n = create_mock_mesh_router(RoutingDirection::N, 0, true);
    auto z_router = create_mock_z_router(100);

    std::map<RoutingDirection, MockRouter*> mesh_routers = {
        {RoutingDirection::N, &mesh_n}
    };

    std::map<RoutingDirection, MockRouter*> z_map = {
        {RoutingDirection::Z, &z_router}
    };

    // Mesh first, then Z
    establish_local_connections(mesh_n, z_map);
    establish_local_connections(z_router, mesh_routers);

    EXPECT_EQ(registry_->size(), 2);

    auto z_to_mesh = registry_->get_connections_by_type(ConnectionType::Z_TO_MESH);
    auto mesh_to_z = registry_->get_connections_by_type(ConnectionType::MESH_TO_Z);

    EXPECT_EQ(z_to_mesh.size(), 1);
    EXPECT_EQ(mesh_to_z.size(), 1);
}
