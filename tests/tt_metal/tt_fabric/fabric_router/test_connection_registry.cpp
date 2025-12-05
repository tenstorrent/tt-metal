// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "tt_metal/fabric/builder/connection_registry.hpp"
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/routing_table_generator.hpp>

namespace tt::tt_fabric {

/**
 * Test fixture for ConnectionRegistry
 * 
 * These tests validate Phase 0 functionality:
 * - Basic registry operations (record, query, clear)
 * - Connection filtering by source, destination, and type
 * - Data structure correctness
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
        .source_sender_channel = 1,
        .dest_node = FabricNodeId(MeshId{0}, 1),
        .dest_direction = RoutingDirection::S,
        .dest_eth_chan = 0,
        .dest_vc = 0,
        .dest_receiver_channel = 1,
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
    EXPECT_EQ(retrieved.source_sender_channel, record.source_sender_channel);
    EXPECT_EQ(retrieved.dest_node, record.dest_node);
    EXPECT_EQ(retrieved.dest_direction, record.dest_direction);
    EXPECT_EQ(retrieved.dest_eth_chan, record.dest_eth_chan);
    EXPECT_EQ(retrieved.dest_vc, record.dest_vc);
    EXPECT_EQ(retrieved.dest_receiver_channel, record.dest_receiver_channel);
    EXPECT_EQ(retrieved.connection_type, record.connection_type);
}

TEST_F(ConnectionRegistryTest, RecordMultipleConnections) {
    // Record 3 different connections
    for (int i = 0; i < 3; ++i) {
        RouterConnectionRecord record{
            .source_node = FabricNodeId(MeshId{0}, i),
            .source_direction = RoutingDirection::N,
            .source_eth_chan = static_cast<uint8_t>(i),
            .source_vc = 0,
            .source_sender_channel = 1,
            .dest_node = FabricNodeId(MeshId{0}, i + 1),
            .dest_direction = RoutingDirection::S,
            .dest_eth_chan = static_cast<uint8_t>(i),
            .dest_vc = 0,
            .dest_receiver_channel = 1,
            .connection_type = ConnectionType::INTRA_MESH
        };
        registry_->record_connection(record);
    }

    EXPECT_EQ(registry_->size(), 3);
    EXPECT_EQ(registry_->get_all_connections().size(), 3);
}

TEST_F(ConnectionRegistryTest, ClearRemovesAllConnections) {
    // Add some connections
    for (int i = 0; i < 5; ++i) {
        RouterConnectionRecord record{
            .source_node = FabricNodeId(MeshId{0}, i),
            .source_direction = RoutingDirection::E,
            .source_eth_chan = static_cast<uint8_t>(i),
            .source_vc = 0,
            .source_sender_channel = 0,
            .dest_node = FabricNodeId(MeshId{0}, i + 1),
            .dest_direction = RoutingDirection::W,
            .dest_eth_chan = static_cast<uint8_t>(i),
            .dest_vc = 0,
            .dest_receiver_channel = 0,
            .connection_type = ConnectionType::INTRA_MESH
        };
        registry_->record_connection(record);
    }

    EXPECT_EQ(registry_->size(), 5);

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
        .source_sender_channel = 1,
        .dest_node = FabricNodeId(MeshId{0}, 1),
        .dest_direction = RoutingDirection::S,
        .dest_eth_chan = 0,
        .dest_vc = 0,
        .dest_receiver_channel = 1,
        .connection_type = ConnectionType::INTRA_MESH
    };
    registry_->record_connection(target);

    // Add noise connection (different source)
    RouterConnectionRecord noise{
        .source_node = FabricNodeId(MeshId{0}, 2),
        .source_direction = RoutingDirection::E,
        .source_eth_chan = 1,
        .source_vc = 0,
        .source_sender_channel = 0,
        .dest_node = FabricNodeId(MeshId{0}, 3),
        .dest_direction = RoutingDirection::W,
        .dest_eth_chan = 1,
        .dest_vc = 0,
        .dest_receiver_channel = 0,
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
    for (int i = 0; i < 3; ++i) {
        RouterConnectionRecord record{
            .source_node = source_node,
            .source_direction = source_dir,
            .source_eth_chan = 0,
            .source_vc = 0,
            .source_sender_channel = static_cast<uint32_t>(i),
            .dest_node = FabricNodeId(MeshId{0}, i + 1),
            .dest_direction = RoutingDirection::S,
            .dest_eth_chan = static_cast<uint8_t>(i),
            .dest_vc = 0,
            .dest_receiver_channel = static_cast<uint32_t>(i),
            .connection_type = ConnectionType::INTRA_MESH
        };
        registry_->record_connection(record);
    }

    auto results = registry_->get_connections_from_source(source_node, source_dir);

    EXPECT_EQ(results.size(), 3);
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
        .source_sender_channel = 1,
        .dest_node = FabricNodeId(MeshId{0}, 1),
        .dest_direction = RoutingDirection::S,
        .dest_eth_chan = 0,
        .dest_vc = 0,
        .dest_receiver_channel = 1,
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
        .source_sender_channel = 1,
        .dest_node = dest_node,
        .dest_direction = dest_dir,
        .dest_eth_chan = 0,
        .dest_vc = 0,
        .dest_receiver_channel = 1,
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
    for (int i = 0; i < 4; ++i) {
        RouterConnectionRecord record{
            .source_node = FabricNodeId(MeshId{0}, i),
            .source_direction = static_cast<RoutingDirection>(i),
            .source_eth_chan = static_cast<uint8_t>(i),
            .source_vc = 1,
            .source_sender_channel = static_cast<uint32_t>(i),
            .dest_node = dest_node,
            .dest_direction = dest_dir,
            .dest_eth_chan = 0,
            .dest_vc = 0,
            .dest_receiver_channel = static_cast<uint32_t>(i),
            .connection_type = ConnectionType::Z_TO_MESH
        };
        registry_->record_connection(record);
    }

    auto results = registry_->get_connections_to_dest(dest_node, dest_dir);

    EXPECT_EQ(results.size(), 4);
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
            .source_sender_channel = 1,
            .dest_node = FabricNodeId(MeshId{0}, i + 1),
            .dest_direction = RoutingDirection::S,
            .dest_eth_chan = static_cast<uint8_t>(i),
            .dest_vc = 0,
            .dest_receiver_channel = 1,
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
        .source_sender_channel = 2,
        .dest_node = FabricNodeId(MeshId{0}, 0),
        .dest_direction = RoutingDirection::Z,
        .dest_eth_chan = 4,
        .dest_vc = 1,
        .dest_receiver_channel = 0,
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
        .source_sender_channel = 1,
        .dest_node = FabricNodeId(MeshId{0}, 1),
        .dest_direction = RoutingDirection::S,
        .dest_eth_chan = 0,
        .dest_vc = 0,
        .dest_receiver_channel = 1,
        .connection_type = ConnectionType::INTRA_MESH
    };
    registry_->record_connection(intra_mesh);

    RouterConnectionRecord mesh_to_z{
        .source_node = FabricNodeId(MeshId{0}, 0),
        .source_direction = RoutingDirection::N,
        .source_eth_chan = 0,
        .source_vc = 0,
        .source_sender_channel = 2,
        .dest_node = FabricNodeId(MeshId{0}, 0),
        .dest_direction = RoutingDirection::Z,
        .dest_eth_chan = 4,
        .dest_vc = 1,
        .dest_receiver_channel = 0,
        .connection_type = ConnectionType::MESH_TO_Z
    };
    registry_->record_connection(mesh_to_z);

    RouterConnectionRecord z_to_mesh{
        .source_node = FabricNodeId(MeshId{0}, 0),
        .source_direction = RoutingDirection::Z,
        .source_eth_chan = 4,
        .source_vc = 1,
        .source_sender_channel = 0,
        .dest_node = FabricNodeId(MeshId{0}, 0),
        .dest_direction = RoutingDirection::N,
        .dest_eth_chan = 0,
        .dest_vc = 0,
        .dest_receiver_channel = 2,
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
        .source_sender_channel = 1,
        .dest_node = FabricNodeId(MeshId{0}, 1),
        .dest_direction = RoutingDirection::S,
        .dest_eth_chan = 0,
        .dest_vc = 0,
        .dest_receiver_channel = 1,
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
            .source_sender_channel = 1,
            .dest_node = FabricNodeId(MeshId{0}, 1),
            .dest_direction = mesh_dirs[(i + 1) % 4],
            .dest_eth_chan = static_cast<uint8_t>((i + 1) % 4),
            .dest_vc = 0,
            .dest_receiver_channel = 1,
            .connection_type = ConnectionType::INTRA_MESH
        };
        registry_->record_connection(record);
    }

    // Add MESH_TO_Z connections (4 mesh routers → Z router)
    for (size_t i = 0; i < 4; ++i) {
        RouterConnectionRecord record{
            .source_node = device_node,
            .source_direction = mesh_dirs[i],
            .source_eth_chan = static_cast<uint8_t>(i),
            .source_vc = 0,
            .source_sender_channel = 2,
            .dest_node = device_node,
            .dest_direction = RoutingDirection::Z,
            .dest_eth_chan = 4,
            .dest_vc = 1,
            .dest_receiver_channel = static_cast<uint32_t>(i),
            .connection_type = ConnectionType::MESH_TO_Z
        };
        registry_->record_connection(record);
    }

    // Add Z_TO_MESH connections (Z router → 4 mesh routers)
    for (size_t i = 0; i < 4; ++i) {
        RouterConnectionRecord record{
            .source_node = device_node,
            .source_direction = RoutingDirection::Z,
            .source_eth_chan = 4,
            .source_vc = 1,
            .source_sender_channel = static_cast<uint32_t>(i),
            .dest_node = device_node,
            .dest_direction = mesh_dirs[i],
            .dest_eth_chan = static_cast<uint8_t>(i),
            .dest_vc = 0,
            .dest_receiver_channel = 2,
            .connection_type = ConnectionType::Z_TO_MESH
        };
        registry_->record_connection(record);
    }

    // Verify totals
    EXPECT_EQ(registry_->size(), 10);  // 2 INTRA_MESH + 4 MESH_TO_Z + 4 Z_TO_MESH

    // Verify by type
    EXPECT_EQ(registry_->get_connections_by_type(ConnectionType::INTRA_MESH).size(), 2);
    EXPECT_EQ(registry_->get_connections_by_type(ConnectionType::MESH_TO_Z).size(), 4);
    EXPECT_EQ(registry_->get_connections_by_type(ConnectionType::Z_TO_MESH).size(), 4);

    // Verify Z router has 4 outgoing connections
    auto z_outgoing = registry_->get_connections_from_source(device_node, RoutingDirection::Z);
    EXPECT_EQ(z_outgoing.size(), 4);

    // Verify Z router has 4 incoming connections
    auto z_incoming = registry_->get_connections_to_dest(device_node, RoutingDirection::Z);
    EXPECT_EQ(z_incoming.size(), 4);
}

// ============ Phase 1.5 Enhancements ============

TEST_F(ConnectionRegistryTest, Phase1_5_MultiTargetReceiver_SameDestination) {
    // Phase 1.5 enables multiple sources connecting to the same destination receiver
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
    
    for (size_t i = 0; i < 4; ++i) {
        RouterConnectionRecord record{
            .source_node = device0,
            .source_direction = mesh_dirs[i],
            .source_eth_chan = static_cast<uint8_t>(i),
            .source_vc = 0,
            .source_sender_channel = 2,
            .dest_node = device0,
            .dest_direction = z_dir,
            .dest_eth_chan = z_eth_chan,
            .dest_vc = z_vc,
            .dest_receiver_channel = z_receiver_ch,  // SAME for all!
            .connection_type = ConnectionType::MESH_TO_Z
        };
        registry_->record_connection(record);
    }
    
    EXPECT_EQ(registry_->size(), 4);
    
    // Query by destination
    auto to_z = registry_->get_connections_to_dest(device0, z_dir);
    ASSERT_EQ(to_z.size(), 4);
    
    // Verify all target the same receiver channel (multi-target)
    for (const auto& conn : to_z) {
        EXPECT_EQ(conn.dest_receiver_channel, z_receiver_ch);
        EXPECT_EQ(conn.dest_vc, z_vc);
    }
    
    // Verify all have different source directions
    std::set<RoutingDirection> sources;
    for (const auto& conn : to_z) {
        sources.insert(conn.source_direction);
    }
    EXPECT_EQ(sources.size(), 4);
}

TEST_F(ConnectionRegistryTest, Phase1_5_QueryMultiTargetByType) {
    // Validate querying multi-target connections by type
    
    FabricNodeId device0(MeshId{0}, 0);
    
    // Add 3 MESH_TO_Z connections (multi-target scenario)
    for (int i = 0; i < 3; ++i) {
        RouterConnectionRecord record{
            .source_node = device0,
            .source_direction = static_cast<RoutingDirection>(i),
            .source_eth_chan = static_cast<uint8_t>(i),
            .source_vc = 0,
            .source_sender_channel = 2,
            .dest_node = device0,
            .dest_direction = RoutingDirection::Z,
            .dest_eth_chan = 4,
            .dest_vc = 1,
            .dest_receiver_channel = 0,  // Same receiver
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
        .source_sender_channel = 1,
        .dest_node = FabricNodeId(MeshId{0}, 1),
        .dest_direction = RoutingDirection::S,
        .dest_eth_chan = 0,
        .dest_vc = 0,
        .dest_receiver_channel = 1,
        .connection_type = ConnectionType::INTRA_MESH
    };
    registry_->record_connection(intra_mesh);
    
    // Query by type
    auto mesh_to_z = registry_->get_connections_by_type(ConnectionType::MESH_TO_Z);
    auto intra = registry_->get_connections_by_type(ConnectionType::INTRA_MESH);
    
    EXPECT_EQ(mesh_to_z.size(), 3);
    EXPECT_EQ(intra.size(), 1);
    
    // Verify MESH_TO_Z connections all target same receiver
    for (const auto& conn : mesh_to_z) {
        EXPECT_EQ(conn.dest_receiver_channel, 0);
    }
}

}  // namespace tt::tt_fabric

