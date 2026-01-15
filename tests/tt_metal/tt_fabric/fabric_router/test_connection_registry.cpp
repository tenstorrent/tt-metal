// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "tt_metal/fabric/builder/connection_registry.hpp"
#include "tt_metal/fabric/builder/router_connection_mapping.hpp"
#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/routing_table_generator.hpp>

namespace tt::tt_fabric {

/**
 * Test fixture for ConnectionRegistry
 *
 * These tests validate core registry functionality:
 * - Basic registry operations (record, query, clear)
 * - Connection filtering by source, destination, and type
 * - Data structure correctness
 *
 * Test Coverage Summary:
 * ┌────────────────────────────────────────────────────────────────────────────────────┐
 * │ Category                 │ Test Name                                 │ Focus        │
 * ├──────────────────────────┼───────────────────────────────────────────┼──────────────┤
 * │ Basic Operations         │ EmptyRegistryHasZeroSize                  │ Empty state  │
 * │                          │ RecordSingleConnection                    │ Single add   │
 * │                          │ RecordMultipleConnections                 │ Multiple add │
 * │                          │ ClearRemovesAllConnections                │ Clear op     │
 * ├──────────────────────────┼───────────────────────────────────────────┼──────────────┤
 * │ Query by Source          │ GetConnectionsFromSource_SingleMatch      │ 1 match      │
 * │                          │ GetConnectionsFromSource_MultipleMatches  │ N matches    │
 * │                          │ GetConnectionsFromSource_NoMatches        │ 0 matches    │
 * ├──────────────────────────┼───────────────────────────────────────────┼──────────────┤
 * │ Query by Destination     │ GetConnectionsToDest_SingleMatch          │ 1 match      │
 * │                          │ GetConnectionsToDest_MultipleMatches      │ N matches    │
 * ├──────────────────────────┼───────────────────────────────────────────┼──────────────┤
 * │ Query by Type            │ GetConnectionsByType_IntraMesh            │ Type filter  │
 * │                          │ GetConnectionsByType_AllTypes             │ All types    │
 * │                          │ GetConnectionsByType_NoMatches            │ Empty result │
 * ├──────────────────────────┼───────────────────────────────────────────┼──────────────┤
 * │ Complex Scenarios        │ ComplexScenario_MixedConnections          │ Full device  │
 * ├──────────────────────────┼───────────────────────────────────────────┼──────────────┤
 * │ Multi-Target Receiver    │ Phase1_5_MultiTargetReceiver_SameDest...  │ Multi-target │
 * │                          │ Phase1_5_QueryMultiTargetByType           │ Type query   │
 * ├──────────────────────────┼───────────────────────────────────────────┼──────────────┤
 * │ Mapping-Driven Conns     │ Phase2_MeshRouter_1D_ConnectionMapping    │ 1D mesh      │
 * │                          │ Phase2_MeshRouter_2D_WithZ_MultipleTargets│ 2D mesh+Z    │
 * │                          │ Phase2_ZRouter_VC1_FourTargets            │ Z router     │
 * │                          │ Phase2_FullDevice_MappingDriven           │ Full device  │
 * │                          │ Phase2_EdgeDevice_2MeshRouters_Mapping... │ Edge device  │
 * └──────────────────────────┴───────────────────────────────────────────┴──────────────┘
 *
 * Total: 20 tests across 6 categories
 */
class ConnectionRegistryTest : public ::testing::Test {
protected:
    void SetUp() override {
        registry_ = std::make_shared<ConnectionRegistry>();
    }

    void TearDown() override {
        registry_.reset();
    }

    std::shared_ptr<ConnectionRegistry> registry_;
};

// ============ Basic Operations Tests ============

TEST_F(ConnectionRegistryTest, EmptyRegistryHasZeroSize) {
    EXPECT_EQ(registry_->size(), 0);
    EXPECT_TRUE(registry_->get_all_connections().empty());
}

TEST_F(ConnectionRegistryTest, RecordSingleConnection) {
    RouterConnectionRecord record{
        .source_node = FabricNodeId(MeshId{0}, 0),
        .source_direction = RoutingDirection::N,
        .source_eth_chan = 0,
        .source_vc = 0,
        .source_receiver_channel = 1,
        .dest_node = FabricNodeId(MeshId{0}, 1),
        .dest_direction = RoutingDirection::S,
        .dest_eth_chan = 0,
        .dest_vc = 0,
        .dest_sender_channel = 1,
        .connection_type = ConnectionType::INTRA_MESH
    };

    registry_->record_connection(record);

    EXPECT_EQ(registry_->size(), 1);

    const auto& connections = registry_->get_all_connections();
    ASSERT_EQ(connections.size(), 1);

    const auto& retrieved = connections[0];
    EXPECT_EQ(retrieved.source_node, record.source_node);
    EXPECT_EQ(retrieved.source_direction, record.source_direction);
    EXPECT_EQ(retrieved.source_eth_chan, record.source_eth_chan);
    EXPECT_EQ(retrieved.source_vc, record.source_vc);
    EXPECT_EQ(retrieved.source_receiver_channel, record.source_receiver_channel);
    EXPECT_EQ(retrieved.dest_node, record.dest_node);
    EXPECT_EQ(retrieved.dest_direction, record.dest_direction);
    EXPECT_EQ(retrieved.dest_eth_chan, record.dest_eth_chan);
    EXPECT_EQ(retrieved.dest_vc, record.dest_vc);
    EXPECT_EQ(retrieved.dest_sender_channel, record.dest_sender_channel);
    EXPECT_EQ(retrieved.connection_type, record.connection_type);
}

TEST_F(ConnectionRegistryTest, RecordMultipleConnections) {
    // Record 3 different connections
    constexpr uint32_t num_test_connections = builder_config::num_downstream_edms_2d_vc0;
    for (uint32_t i = 0; i < num_test_connections; ++i) {
        RouterConnectionRecord record{
            .source_node = FabricNodeId(MeshId{0}, i),
            .source_direction = RoutingDirection::N,
            .source_eth_chan = static_cast<uint8_t>(i),
            .source_vc = 0,
            .source_receiver_channel = 1,
            .dest_node = FabricNodeId(MeshId{0}, i + 1),
            .dest_direction = RoutingDirection::S,
            .dest_eth_chan = static_cast<uint8_t>(i),
            .dest_vc = 0,
            .dest_sender_channel = 1,
            .connection_type = ConnectionType::INTRA_MESH
        };
        registry_->record_connection(record);
    }

    EXPECT_EQ(registry_->size(), num_test_connections);
    EXPECT_EQ(registry_->get_all_connections().size(), num_test_connections);
}

TEST_F(ConnectionRegistryTest, ClearRemovesAllConnections) {
    // Add some connections
    constexpr uint32_t num_test_connections = 5;
    for (uint32_t i = 0; i < num_test_connections; ++i) {
        RouterConnectionRecord record{
            .source_node = FabricNodeId(MeshId{0}, i),
            .source_direction = RoutingDirection::E,
            .source_eth_chan = static_cast<uint8_t>(i),
            .source_vc = 0,
            .source_receiver_channel = 0,
            .dest_node = FabricNodeId(MeshId{0}, i + 1),
            .dest_direction = RoutingDirection::W,
            .dest_eth_chan = static_cast<uint8_t>(i),
            .dest_vc = 0,
            .dest_sender_channel = 0,
            .connection_type = ConnectionType::INTRA_MESH
        };
        registry_->record_connection(record);
    }

    EXPECT_EQ(registry_->size(), num_test_connections);

    registry_->clear();

    EXPECT_EQ(registry_->size(), 0);
    EXPECT_TRUE(registry_->get_all_connections().empty());
}

// ============ Query by Source Tests ============

TEST_F(ConnectionRegistryTest, GetConnectionsFromSource_SingleMatch) {
    FabricNodeId source_node(MeshId{0}, 0);
    RoutingDirection source_dir = RoutingDirection::N;

    // Add target connection
    RouterConnectionRecord target{
        .source_node = source_node,
        .source_direction = source_dir,
        .source_eth_chan = 0,
        .source_vc = 0,
        .source_receiver_channel = 1,
        .dest_node = FabricNodeId(MeshId{0}, 1),
        .dest_direction = RoutingDirection::S,
        .dest_eth_chan = 0,
        .dest_vc = 0,
        .dest_sender_channel = 1,
        .connection_type = ConnectionType::INTRA_MESH
    };
    registry_->record_connection(target);

    // Add noise connection (different source)
    RouterConnectionRecord noise{
        .source_node = FabricNodeId(MeshId{0}, 2),
        .source_direction = RoutingDirection::E,
        .source_eth_chan = 1,
        .source_vc = 0,
        .source_receiver_channel = 0,
        .dest_node = FabricNodeId(MeshId{0}, 3),
        .dest_direction = RoutingDirection::W,
        .dest_eth_chan = 1,
        .dest_vc = 0,
        .dest_sender_channel = 0,
        .connection_type = ConnectionType::INTRA_MESH
    };
    registry_->record_connection(noise);

    auto results = registry_->get_connections_from_source(source_node, source_dir);

    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].source_node, source_node);
    EXPECT_EQ(results[0].source_direction, source_dir);
}

TEST_F(ConnectionRegistryTest, GetConnectionsFromSource_MultipleMatches) {
    FabricNodeId source_node(MeshId{0}, 0);
    RoutingDirection source_dir = RoutingDirection::N;

    // Add 3 connections from same source
    constexpr uint32_t num_connections = builder_config::num_downstream_edms_2d_vc0;
    for (uint32_t i = 0; i < num_connections; ++i) {
        RouterConnectionRecord record{
            .source_node = source_node,
            .source_direction = source_dir,
            .source_eth_chan = 0,
            .source_vc = 0,
            .source_receiver_channel = static_cast<uint32_t>(i),
            .dest_node = FabricNodeId(MeshId{0}, i + 1),
            .dest_direction = RoutingDirection::S,
            .dest_eth_chan = static_cast<uint8_t>(i),
            .dest_vc = 0,
            .dest_sender_channel = static_cast<uint32_t>(i),
            .connection_type = ConnectionType::INTRA_MESH
        };
        registry_->record_connection(record);
    }

    auto results = registry_->get_connections_from_source(source_node, source_dir);

    EXPECT_EQ(results.size(), num_connections);
    for (const auto& conn : results) {
        EXPECT_EQ(conn.source_node, source_node);
        EXPECT_EQ(conn.source_direction, source_dir);
    }
}

TEST_F(ConnectionRegistryTest, GetConnectionsFromSource_NoMatches) {
    // Add a connection
    RouterConnectionRecord record{
        .source_node = FabricNodeId(MeshId{0}, 0),
        .source_direction = RoutingDirection::N,
        .source_eth_chan = 0,
        .source_vc = 0,
        .source_receiver_channel = 1,
        .dest_node = FabricNodeId(MeshId{0}, 1),
        .dest_direction = RoutingDirection::S,
        .dest_eth_chan = 0,
        .dest_vc = 0,
        .dest_sender_channel = 1,
        .connection_type = ConnectionType::INTRA_MESH
    };
    registry_->record_connection(record);

    // Query for non-existent source
    auto results = registry_->get_connections_from_source(
        FabricNodeId(MeshId{0}, 99), RoutingDirection::E);

    EXPECT_TRUE(results.empty());
}

// ============ Query by Destination Tests ============

TEST_F(ConnectionRegistryTest, GetConnectionsToDest_SingleMatch) {
    FabricNodeId dest_node(MeshId{0}, 1);
    RoutingDirection dest_dir = RoutingDirection::S;

    RouterConnectionRecord target{
        .source_node = FabricNodeId(MeshId{0}, 0),
        .source_direction = RoutingDirection::N,
        .source_eth_chan = 0,
        .source_vc = 0,
        .source_receiver_channel = 1,
        .dest_node = dest_node,
        .dest_direction = dest_dir,
        .dest_eth_chan = 0,
        .dest_vc = 0,
        .dest_sender_channel = 1,
        .connection_type = ConnectionType::INTRA_MESH
    };
    registry_->record_connection(target);

    auto results = registry_->get_connections_to_dest(dest_node, dest_dir);

    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].dest_node, dest_node);
    EXPECT_EQ(results[0].dest_direction, dest_dir);
}

TEST_F(ConnectionRegistryTest, GetConnectionsToDest_MultipleMatches) {
    FabricNodeId dest_node(MeshId{0}, 5);
    RoutingDirection dest_dir = RoutingDirection::W;

    // Add 4 connections to same destination (simulating Z router scenario)
    constexpr uint32_t num_z_connections = builder_config::num_mesh_directions_2d;
    for (uint32_t i = 0; i < num_z_connections; ++i) {
        RouterConnectionRecord record{
            .source_node = FabricNodeId(MeshId{0}, i),
            .source_direction = static_cast<RoutingDirection>(i),
            .source_eth_chan = static_cast<uint8_t>(i),
            .source_vc = 1,
            .source_receiver_channel = static_cast<uint32_t>(i),
            .dest_node = dest_node,
            .dest_direction = dest_dir,
            .dest_eth_chan = 0,
            .dest_vc = 0,
            .dest_sender_channel = static_cast<uint32_t>(i),
            .connection_type = ConnectionType::Z_TO_MESH
        };
        registry_->record_connection(record);
    }

    auto results = registry_->get_connections_to_dest(dest_node, dest_dir);

    EXPECT_EQ(results.size(), num_z_connections);
    for (const auto& conn : results) {
        EXPECT_EQ(conn.dest_node, dest_node);
        EXPECT_EQ(conn.dest_direction, dest_dir);
    }
}

// ============ Query by Connection Type Tests ============

TEST_F(ConnectionRegistryTest, GetConnectionsByType_IntraMesh) {
    // Add 2 INTRA_MESH connections
    for (int i = 0; i < 2; ++i) {
        RouterConnectionRecord record{
            .source_node = FabricNodeId(MeshId{0}, i),
            .source_direction = RoutingDirection::N,
            .source_eth_chan = static_cast<uint8_t>(i),
            .source_vc = 0,
            .source_receiver_channel = 1,
            .dest_node = FabricNodeId(MeshId{0}, i + 1),
            .dest_direction = RoutingDirection::S,
            .dest_eth_chan = static_cast<uint8_t>(i),
            .dest_vc = 0,
            .dest_sender_channel = 1,
            .connection_type = ConnectionType::INTRA_MESH
        };
        registry_->record_connection(record);
    }

    // Add 1 MESH_TO_Z connection
    RouterConnectionRecord mesh_to_z{
        .source_node = FabricNodeId(MeshId{0}, 0),
        .source_direction = RoutingDirection::N,
        .source_eth_chan = 0,
        .source_vc = 0,
        .source_receiver_channel = 2,
        .dest_node = FabricNodeId(MeshId{0}, 0),
        .dest_direction = RoutingDirection::Z,
        .dest_eth_chan = 4,
        .dest_vc = 1,
        .dest_sender_channel = 0,
        .connection_type = ConnectionType::MESH_TO_Z
    };
    registry_->record_connection(mesh_to_z);

    auto intra_mesh_results = registry_->get_connections_by_type(ConnectionType::INTRA_MESH);
    auto mesh_to_z_results = registry_->get_connections_by_type(ConnectionType::MESH_TO_Z);

    EXPECT_EQ(intra_mesh_results.size(), 2);
    EXPECT_EQ(mesh_to_z_results.size(), 1);

    for (const auto& conn : intra_mesh_results) {
        EXPECT_EQ(conn.connection_type, ConnectionType::INTRA_MESH);
    }

    EXPECT_EQ(mesh_to_z_results[0].connection_type, ConnectionType::MESH_TO_Z);
}

TEST_F(ConnectionRegistryTest, GetConnectionsByType_AllTypes) {
    // Add one of each connection type
    RouterConnectionRecord intra_mesh{
        .source_node = FabricNodeId(MeshId{0}, 0),
        .source_direction = RoutingDirection::N,
        .source_eth_chan = 0,
        .source_vc = 0,
        .source_receiver_channel = 1,
        .dest_node = FabricNodeId(MeshId{0}, 1),
        .dest_direction = RoutingDirection::S,
        .dest_eth_chan = 0,
        .dest_vc = 0,
        .dest_sender_channel = 1,
        .connection_type = ConnectionType::INTRA_MESH
    };
    registry_->record_connection(intra_mesh);

    RouterConnectionRecord mesh_to_z{
        .source_node = FabricNodeId(MeshId{0}, 0),
        .source_direction = RoutingDirection::N,
        .source_eth_chan = 0,
        .source_vc = 0,
        .source_receiver_channel = 2,
        .dest_node = FabricNodeId(MeshId{0}, 0),
        .dest_direction = RoutingDirection::Z,
        .dest_eth_chan = 4,
        .dest_vc = 1,
        .dest_sender_channel = 0,
        .connection_type = ConnectionType::MESH_TO_Z
    };
    registry_->record_connection(mesh_to_z);

    RouterConnectionRecord z_to_mesh{
        .source_node = FabricNodeId(MeshId{0}, 0),
        .source_direction = RoutingDirection::Z,
        .source_eth_chan = 4,
        .source_vc = 1,
        .source_receiver_channel = 0,
        .dest_node = FabricNodeId(MeshId{0}, 0),
        .dest_direction = RoutingDirection::N,
        .dest_eth_chan = 0,
        .dest_vc = 0,
        .dest_sender_channel = 2,
        .connection_type = ConnectionType::Z_TO_MESH
    };
    registry_->record_connection(z_to_mesh);

    EXPECT_EQ(registry_->size(), 3);
    EXPECT_EQ(registry_->get_connections_by_type(ConnectionType::INTRA_MESH).size(), 1);
    EXPECT_EQ(registry_->get_connections_by_type(ConnectionType::MESH_TO_Z).size(), 1);
    EXPECT_EQ(registry_->get_connections_by_type(ConnectionType::Z_TO_MESH).size(), 1);
}

TEST_F(ConnectionRegistryTest, GetConnectionsByType_NoMatches) {
    // Add only INTRA_MESH connections
    RouterConnectionRecord record{
        .source_node = FabricNodeId(MeshId{0}, 0),
        .source_direction = RoutingDirection::N,
        .source_eth_chan = 0,
        .source_vc = 0,
        .source_receiver_channel = 1,
        .dest_node = FabricNodeId(MeshId{0}, 1),
        .dest_direction = RoutingDirection::S,
        .dest_eth_chan = 0,
        .dest_vc = 0,
        .dest_sender_channel = 1,
        .connection_type = ConnectionType::INTRA_MESH
    };
    registry_->record_connection(record);

    // Query for Z_TO_MESH (should be empty)
    auto results = registry_->get_connections_by_type(ConnectionType::Z_TO_MESH);
    EXPECT_TRUE(results.empty());
}

// ============ Complex Scenario Tests ============

TEST_F(ConnectionRegistryTest, ComplexScenario_MixedConnections) {
    // Simulate a device with 4 mesh routers and 1 Z router
    // Each mesh router connects to 2 other mesh routers (INTRA_MESH)
    // Each mesh router connects to Z router (MESH_TO_Z)
    // Z router connects to all 4 mesh routers (Z_TO_MESH)

    FabricNodeId device_node(MeshId{0}, 0);

    // 4 mesh routers: N, E, S, W
    std::vector<RoutingDirection> mesh_dirs = {
        RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W
    };

    // Add INTRA_MESH connections (simplified - just a few)
    for (size_t i = 0; i < 2; ++i) {
        RouterConnectionRecord record{
            .source_node = device_node,
            .source_direction = mesh_dirs[i],
            .source_eth_chan = static_cast<uint8_t>(i),
            .source_vc = 0,
            .source_receiver_channel = 1,
            .dest_node = FabricNodeId(MeshId{0}, 1),
            .dest_direction = mesh_dirs[(i + 1) % 4],
            .dest_eth_chan = static_cast<uint8_t>((i + 1) % 4),
            .dest_vc = 0,
            .dest_sender_channel = 1,
            .connection_type = ConnectionType::INTRA_MESH
        };
        registry_->record_connection(record);
    }

    // Add MESH_TO_Z connections (4 mesh routers → Z router)
    constexpr uint32_t num_mesh_routers = builder_config::num_mesh_directions_2d;
    for (size_t i = 0; i < num_mesh_routers; ++i) {
        RouterConnectionRecord record{
            .source_node = device_node,
            .source_direction = mesh_dirs[i],
            .source_eth_chan = static_cast<uint8_t>(i),
            .source_vc = 0,
            .source_receiver_channel = 2,
            .dest_node = device_node,
            .dest_direction = RoutingDirection::Z,
            .dest_eth_chan = 4,
            .dest_vc = 1,
            .dest_sender_channel = static_cast<uint32_t>(i),
            .connection_type = ConnectionType::MESH_TO_Z
        };
        registry_->record_connection(record);
    }

    // Add Z_TO_MESH connections (Z router → 4 mesh routers)
    for (size_t i = 0; i < num_mesh_routers; ++i) {
        RouterConnectionRecord record{
            .source_node = device_node,
            .source_direction = RoutingDirection::Z,
            .source_eth_chan = 4,
            .source_vc = 1,
            .source_receiver_channel = static_cast<uint32_t>(i),
            .dest_node = device_node,
            .dest_direction = mesh_dirs[i],
            .dest_eth_chan = static_cast<uint8_t>(i),
            .dest_vc = 0,
            .dest_sender_channel = 2,
            .connection_type = ConnectionType::Z_TO_MESH
        };
        registry_->record_connection(record);
    }

    // Verify totals
    constexpr uint32_t num_intra_mesh = 2;
    EXPECT_EQ(registry_->size(), num_intra_mesh + (2 * num_mesh_routers));  // 2 INTRA_MESH + 4 MESH_TO_Z + 4 Z_TO_MESH

    // Verify by type
    EXPECT_EQ(registry_->get_connections_by_type(ConnectionType::INTRA_MESH).size(), num_intra_mesh);
    EXPECT_EQ(registry_->get_connections_by_type(ConnectionType::MESH_TO_Z).size(), num_mesh_routers);
    EXPECT_EQ(registry_->get_connections_by_type(ConnectionType::Z_TO_MESH).size(), num_mesh_routers);

    // Verify Z router has 4 outgoing connections
    auto z_outgoing = registry_->get_connections_from_source(device_node, RoutingDirection::Z);
    EXPECT_EQ(z_outgoing.size(), num_mesh_routers);

    // Verify Z router has 4 incoming connections
    auto z_incoming = registry_->get_connections_to_dest(device_node, RoutingDirection::Z);
    EXPECT_EQ(z_incoming.size(), num_mesh_routers);
}

// ============ Multi-Target Receiver Tests ============

TEST_F(ConnectionRegistryTest, Phase1_5_MultiTargetReceiver_SameDestination) {
    // Multi-target receiver: multiple sources connecting to the same destination receiver
    // This test validates that the registry correctly tracks this scenario

    FabricNodeId device0(MeshId{0}, 0);
    RoutingDirection z_dir = RoutingDirection::Z;
    uint8_t z_eth_chan = 4;
    uint32_t z_vc = 1;
    uint32_t z_receiver_ch = 0;  // SAME receiver for all connections

    // 4 mesh routers all connect to the same Z router receiver
    std::vector<RoutingDirection> mesh_dirs = {
        RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W
    };

    constexpr uint32_t num_mesh_routers = builder_config::num_mesh_directions_2d;
    for (size_t i = 0; i < num_mesh_routers; ++i) {
        RouterConnectionRecord record{
            .source_node = device0,
            .source_direction = mesh_dirs[i],
            .source_eth_chan = static_cast<uint8_t>(i),
            .source_vc = 0,
            .source_receiver_channel = 2,
            .dest_node = device0,
            .dest_direction = z_dir,
            .dest_eth_chan = z_eth_chan,
            .dest_vc = z_vc,
            .dest_sender_channel = z_receiver_ch,  // SAME for all!
            .connection_type = ConnectionType::MESH_TO_Z
        };
        registry_->record_connection(record);
    }

    EXPECT_EQ(registry_->size(), num_mesh_routers);

    // Query by destination
    auto to_z = registry_->get_connections_to_dest(device0, z_dir);
    ASSERT_EQ(to_z.size(), num_mesh_routers);

    // Verify all target the same receiver channel (multi-target)
    for (const auto& conn : to_z) {
        EXPECT_EQ(conn.dest_sender_channel, z_receiver_ch);
        EXPECT_EQ(conn.dest_vc, z_vc);
    }

    // Verify all have different source directions
    std::set<RoutingDirection> sources;
    for (const auto& conn : to_z) {
        sources.insert(conn.source_direction);
    }
    EXPECT_EQ(sources.size(), num_mesh_routers);
}

TEST_F(ConnectionRegistryTest, Phase1_5_QueryMultiTargetByType) {
    // Validate querying multi-target connections by type

    FabricNodeId device0(MeshId{0}, 0);

    // Add 3 MESH_TO_Z connections (multi-target scenario)
    constexpr uint32_t num_connections = builder_config::num_downstream_edms_2d_vc0;
    for (uint32_t i = 0; i < num_connections; ++i) {
        RouterConnectionRecord record{
            .source_node = device0,
            .source_direction = static_cast<RoutingDirection>(i),
            .source_eth_chan = static_cast<uint8_t>(i),
            .source_vc = 0,
            .source_receiver_channel = 2,
            .dest_node = device0,
            .dest_direction = RoutingDirection::Z,
            .dest_eth_chan = 4,
            .dest_vc = 1,
            .dest_sender_channel = 0,  // Same receiver
            .connection_type = ConnectionType::MESH_TO_Z
        };
        registry_->record_connection(record);
    }

    // Add 1 INTRA_MESH connection (single target)
    RouterConnectionRecord intra_mesh{
        .source_node = device0,
        .source_direction = RoutingDirection::N,
        .source_eth_chan = 0,
        .source_vc = 0,
        .source_receiver_channel = 1,
        .dest_node = FabricNodeId(MeshId{0}, 1),
        .dest_direction = RoutingDirection::S,
        .dest_eth_chan = 0,
        .dest_vc = 0,
        .dest_sender_channel = 1,
        .connection_type = ConnectionType::INTRA_MESH
    };
    registry_->record_connection(intra_mesh);

    // Query by type
    auto mesh_to_z = registry_->get_connections_by_type(ConnectionType::MESH_TO_Z);
    auto intra = registry_->get_connections_by_type(ConnectionType::INTRA_MESH);

    EXPECT_EQ(mesh_to_z.size(), num_connections);
    EXPECT_EQ(intra.size(), 1);

    // Verify MESH_TO_Z connections all target same receiver
    for (const auto& conn : mesh_to_z) {
        EXPECT_EQ(conn.dest_sender_channel, 0);
    }
}

// ============ Mapping-Driven Connection Tests ============

TEST_F(ConnectionRegistryTest, Phase2_MeshRouter_1D_ConnectionMapping) {
    // Create connection mapping for 1D mesh router
    auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Linear,
        RoutingDirection::N,
        false);  // No Z router

    // Verify mapping has expected targets
    constexpr uint32_t receiver_channel = 0;
    auto targets = mapping.get_downstream_targets(0, receiver_channel);
    ASSERT_EQ(targets.size(), 1);
    EXPECT_EQ(targets[0].type, ConnectionType::INTRA_MESH);
    EXPECT_EQ(targets[0].target_direction.value(), RoutingDirection::S);

    // Simulate recording connections based on mapping
    FabricNodeId source(MeshId{0}, 0);
    FabricNodeId dest(MeshId{0}, 1);

    for (const auto& target : targets) {
        RouterConnectionRecord record{
            .source_node = source,
            .source_direction = RoutingDirection::N,
            .source_eth_chan = 0,
            .source_vc = 0,
            .source_receiver_channel = receiver_channel,
            .dest_node = dest,
            .dest_direction = target.target_direction.value(),
            .dest_eth_chan = 0,
            .dest_vc = target.target_vc,
            .dest_sender_channel = 0,
            .connection_type = target.type
        };
        registry_->record_connection(record);
    }

    EXPECT_EQ(registry_->size(), 1);

    auto intra_mesh = registry_->get_connections_by_type(ConnectionType::INTRA_MESH);
    EXPECT_EQ(intra_mesh.size(), 1);
}

TEST_F(ConnectionRegistryTest, Phase2_MeshRouter_2D_WithZ_MultipleTargets) {
    // Create connection mapping for 2D mesh router with Z
    auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh,
        RoutingDirection::N,
        true);  // Has Z router

    // Verify mapping: receiver channel 0 has 4 downstream targets (3 INTRA_MESH + 1 MESH_TO_Z)
    constexpr uint32_t num_intra_mesh_targets = builder_config::num_downstream_edms_2d_vc0;
    constexpr uint32_t num_mesh_to_z_targets = 1;
    constexpr uint32_t total_targets = num_intra_mesh_targets + num_mesh_to_z_targets;

    auto targets = mapping.get_downstream_targets(0, 0);  // VC0, receiver channel 0
    ASSERT_EQ(targets.size(), total_targets);

    // Verify target types
    uint32_t intra_mesh_count = 0;
    uint32_t mesh_to_z_count = 0;
    for (const auto& target : targets) {
        if (target.type == ConnectionType::INTRA_MESH) {
            intra_mesh_count++;
        } else if (target.type == ConnectionType::MESH_TO_Z) {
            mesh_to_z_count++;
        }
    }
    EXPECT_EQ(intra_mesh_count, num_intra_mesh_targets);
    EXPECT_EQ(mesh_to_z_count, num_mesh_to_z_targets);

    // Simulate recording all connections from receiver channel 0
    FabricNodeId source(MeshId{0}, 0);

    for (size_t i = 0; i < targets.size(); ++i) {
        const auto& target = targets[i];
        RouterConnectionRecord record{
            .source_node = source,
            .source_direction = RoutingDirection::N,
            .source_eth_chan = 0,
            .source_vc = 0,
            .source_receiver_channel = 0,
            .dest_node = (target.type == ConnectionType::MESH_TO_Z) ? FabricNodeId(MeshId{0}, 100)
                                                                    : FabricNodeId(MeshId{0}, static_cast<uint32_t>(i)),
            .dest_direction = target.target_direction.value(),
            .dest_eth_chan = 0,
            .dest_vc = target.target_vc,
            .dest_sender_channel = target.target_sender_channel,
            .connection_type = target.type};
        registry_->record_connection(record);
    }

    // Verify registry has all connections
    EXPECT_EQ(registry_->size(), total_targets);

    auto intra_mesh = registry_->get_connections_by_type(ConnectionType::INTRA_MESH);
    auto mesh_to_z = registry_->get_connections_by_type(ConnectionType::MESH_TO_Z);

    EXPECT_EQ(intra_mesh.size(), num_intra_mesh_targets);
    EXPECT_EQ(mesh_to_z.size(), num_mesh_to_z_targets);
}

TEST_F(ConnectionRegistryTest, Phase2_ZRouter_VC1_FourTargets) {
    // Create connection mapping for Z router
    auto mapping = RouterConnectionMapping::for_z_router();

    // Verify mapping: receiver channel 0 on VC1 has 4 downstream targets
    constexpr uint32_t num_z_targets = builder_config::num_sender_channels_z_router_vc1;

    auto targets = mapping.get_downstream_targets(1, 0);  // VC1, receiver channel 0
    ASSERT_EQ(targets.size(), num_z_targets);

    // Verify all targets are Z_TO_MESH with correct directions
    std::vector<RoutingDirection> expected_directions = {
        RoutingDirection::N,
        RoutingDirection::E,
        RoutingDirection::S,
        RoutingDirection::W
    };

    for (size_t i = 0; i < targets.size(); ++i) {
        EXPECT_EQ(targets[i].type, ConnectionType::Z_TO_MESH);
        EXPECT_EQ(targets[i].target_direction.value(), expected_directions[i]);
    }

    // Simulate recording all Z→mesh connections from receiver channel 0
    FabricNodeId z_node(MeshId{0}, 100);

    for (size_t i = 0; i < targets.size(); ++i) {
        const auto& target = targets[i];
        RouterConnectionRecord record{
            .source_node = z_node,
            .source_direction = RoutingDirection::Z,
            .source_eth_chan = 0,
            .source_vc = 1,
            .source_receiver_channel = 0,  // All from receiver channel 0
            .dest_node = FabricNodeId(MeshId{0}, static_cast<uint32_t>(i)),
            .dest_direction = target.target_direction.value(),
            .dest_eth_chan = 0,
            .dest_vc = target.target_vc,
            .dest_sender_channel = target.target_sender_channel,
            .connection_type = ConnectionType::Z_TO_MESH
        };
        registry_->record_connection(record);
    }

    // Verify registry has all connections
    EXPECT_EQ(registry_->size(), num_z_targets);

    auto z_to_mesh = registry_->get_connections_by_type(ConnectionType::Z_TO_MESH);
    EXPECT_EQ(z_to_mesh.size(), num_z_targets);

    // Verify all connections use VC1 and target VC1
    for (const auto& conn : z_to_mesh) {
        EXPECT_EQ(conn.source_vc, 1);
        EXPECT_EQ(conn.dest_vc, 1);  // Target mesh router VC1
    }
}

TEST_F(ConnectionRegistryTest, Phase2_FullDevice_MappingDriven) {
    // Simulate a full device with 4 mesh routers + 1 Z router
    // All connections driven by connection mappings

    std::vector<RoutingDirection> mesh_directions = {
        RoutingDirection::N,
        RoutingDirection::E,
        RoutingDirection::S,
        RoutingDirection::W
    };

    // Create mesh router mappings
    for (size_t i = 0; i < 4; ++i) {
        auto mapping = RouterConnectionMapping::for_mesh_router(
            Topology::Mesh,
            mesh_directions[i],
            true);  // Has Z router

        FabricNodeId source(MeshId{0}, i);

        // Get all targets from receiver channel 0
        auto targets = mapping.get_downstream_targets(0, 0);

        // Record all connections from receiver channel 0
        for (size_t t = 0; t < targets.size(); ++t) {
            const auto& target = targets[t];
            RouterConnectionRecord record{
                .source_node = source,
                .source_direction = mesh_directions[i],
                .source_eth_chan = static_cast<uint8_t>(i),
                .source_vc = 0,
                .source_receiver_channel = 0,
                .dest_node = (target.type == ConnectionType::MESH_TO_Z) ?
                             FabricNodeId(MeshId{0}, 100) : FabricNodeId(MeshId{0}, (i + t) % 4),
                .dest_direction = target.target_direction.value(),
                .dest_eth_chan = 0,
                .dest_vc = target.target_vc,
                .dest_sender_channel = target.target_sender_channel,
                .connection_type = target.type
            };
            registry_->record_connection(record);
        }
    }

    // Create Z router mapping and record connections
    auto z_mapping = RouterConnectionMapping::for_z_router();
    FabricNodeId z_node(MeshId{0}, 100);

    // Get all targets from receiver channel 0 on VC1
    auto z_targets = z_mapping.get_downstream_targets(1, 0);
    ASSERT_EQ(z_targets.size(), 4);

    for (size_t i = 0; i < z_targets.size(); ++i) {
        const auto& target = z_targets[i];
        RouterConnectionRecord record{
            .source_node = z_node,
            .source_direction = RoutingDirection::Z,
            .source_eth_chan = 0,
            .source_vc = 1,
            .source_receiver_channel = 0,  // All from receiver channel 0
            .dest_node = FabricNodeId(MeshId{0}, static_cast<uint32_t>(i)),
            .dest_direction = target.target_direction.value(),
            .dest_eth_chan = 0,
            .dest_vc = target.target_vc,
            .dest_sender_channel = target.target_sender_channel,
            .connection_type = ConnectionType::Z_TO_MESH
        };
        registry_->record_connection(record);
    }

    // Verify total connections: 4 routers × 4 connections + 4 Z connections = 20
    EXPECT_EQ(registry_->size(), 20);

    // Verify connection type distribution
    auto intra_mesh = registry_->get_connections_by_type(ConnectionType::INTRA_MESH);
    auto mesh_to_z = registry_->get_connections_by_type(ConnectionType::MESH_TO_Z);
    auto z_to_mesh = registry_->get_connections_by_type(ConnectionType::Z_TO_MESH);

    EXPECT_EQ(intra_mesh.size(), 12);  // 4 routers × 3 INTRA_MESH
    EXPECT_EQ(mesh_to_z.size(), 4);    // 4 routers × 1 MESH_TO_Z
    EXPECT_EQ(z_to_mesh.size(), 4);    // Z router × 4 Z_TO_MESH
}

TEST_F(ConnectionRegistryTest, Phase2_EdgeDevice_2MeshRouters_MappingDriven) {
    // Edge device with only 2 mesh routers (NORTH and EAST) + Z router
    // Z router mapping specifies all 4 directions, but only 2 exist

    std::vector<RoutingDirection> mesh_directions = {
        RoutingDirection::N,
        RoutingDirection::E
    };

    // Create mesh router mappings and record connections
    for (size_t i = 0; i < 2; ++i) {
        auto mapping = RouterConnectionMapping::for_mesh_router(
            Topology::Mesh,
            mesh_directions[i],
            true);  // Has Z router

        FabricNodeId source(MeshId{0}, i);

        // Get all targets from receiver channel 0
        auto targets = mapping.get_downstream_targets(0, 0);

        // Find and record MESH_TO_Z connection
        for (const auto& target : targets) {
            if (target.type == ConnectionType::MESH_TO_Z) {
                RouterConnectionRecord z_record{
                    .source_node = source,
                    .source_direction = mesh_directions[i],
                    .source_eth_chan = static_cast<uint8_t>(i),
                    .source_vc = 0,
                    .source_receiver_channel = 0,
                    .dest_node = FabricNodeId(MeshId{0}, 100),
                    .dest_direction = RoutingDirection::Z,
                    .dest_eth_chan = 0,
                    .dest_vc = target.target_vc,
                    .dest_sender_channel = target.target_sender_channel,
                    .connection_type = ConnectionType::MESH_TO_Z
                };
                registry_->record_connection(z_record);
            }
        }
    }

    // Create Z router mapping
    auto z_mapping = RouterConnectionMapping::for_z_router();
    FabricNodeId z_node(MeshId{0}, 100);

    // Z router has intent for all 4 directions, but only 2 exist
    // Simulate FabricBuilder: only record connections for existing routers
    std::set<RoutingDirection> existing_routers = {
        RoutingDirection::N,
        RoutingDirection::E
    };

    // Get all targets from receiver channel 0 on VC1
    auto z_targets = z_mapping.get_downstream_targets(1, 0);
    ASSERT_EQ(z_targets.size(), 4);

    // Only record connections for existing routers
    for (const auto& target : z_targets) {
        if (existing_routers.contains(target.target_direction.value())) {
            RouterConnectionRecord record{
                .source_node = z_node,
                .source_direction = RoutingDirection::Z,
                .source_eth_chan = 0,
                .source_vc = 1,
                .source_receiver_channel = 0,  // All from receiver channel 0
                .dest_node = FabricNodeId(MeshId{0}, 0),  // Simplified for test
                .dest_direction = target.target_direction.value(),
                .dest_eth_chan = 0,
                .dest_vc = target.target_vc,
                .dest_sender_channel = target.target_sender_channel,
                .connection_type = ConnectionType::Z_TO_MESH
            };
            registry_->record_connection(record);
        }
    }

    // Verify connections: 2 MESH_TO_Z + 2 Z_TO_MESH = 4
    EXPECT_EQ(registry_->size(), 4);

    auto mesh_to_z = registry_->get_connections_by_type(ConnectionType::MESH_TO_Z);
    auto z_to_mesh = registry_->get_connections_by_type(ConnectionType::Z_TO_MESH);

    EXPECT_EQ(mesh_to_z.size(), 2);
    EXPECT_EQ(z_to_mesh.size(), 2);
}

}  // namespace tt::tt_fabric
