// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "tt_metal/fabric/builder/connection_registry.hpp"
#include "tt_metal/fabric/builder/router_connection_mapping.hpp"
#include "tt_metal/fabric/fabric_router_channel_mapping.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/builder/mesh_channel_spec.hpp"
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/routing_table_generator.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace tt::tt_fabric {

/**
 * Test fixture for router connection scenarios
 *
 * These tests validate router connection integration:
 * - Non-Z router connecting to non-Z router (INTRA_MESH)
 * - Non-Z router connecting to Z router (MESH_TO_Z)
 * - Z router connecting to non-Z router (Z_TO_MESH)
 * - Channel mapping correctness for each scenario
 *
 * Test Coverage Summary:
 * ┌──────────────────────────────────────────────────────────────────────────────────────┐
 * │ Category                    │ Test Name                                │ Focus        │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Mesh↔Mesh (INTRA_MESH)      │ MeshToMesh_VC0_Connection                │ Basic mesh   │
 * │                             │ MeshToMesh_1D_Linear                     │ 1D topology  │
 * │                             │ MeshToMesh_2D_AllDirections              │ All dirs     │
 * │                             │ MeshToMesh_MultipleDevices               │ Multi-device │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Mesh→Z (MESH_TO_Z)          │ MeshToZ_VC0_Connection                   │ Basic M→Z    │
 * │                             │ MeshToZ_AllMeshRouters_ToSameZ           │ 4→1          │
 * │                             │ MeshToZ_ChannelMapping_Correct           │ Channels     │
 * │                             │ MeshToZ_MultiTargetReceiver              │ Multi-target │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Z→Mesh (Z_TO_MESH)          │ ZToMesh_VC1_Connection                   │ Basic Z→M    │
 * │                             │ ZToMesh_FourDirections                   │ 4 directions │
 * │                             │ ZToMesh_ChannelMapping_Correct           │ Channels     │
 * │                             │ ZToMesh_TargetVC_Correct                 │ Target VC    │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Full Device Integration     │ FullDevice_4Mesh1Z_AllConnectionTypes    │ All types    │
 * │                             │ FullDevice_4Mesh1Z_VerifyRegistry        │ Registry     │
 * │                             │ FullDevice_4Mesh1Z_ConnectionCounts      │ Counts       │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Edge Device Scenarios       │ EdgeDevice_2Mesh1Z_PartialConnections    │ 2 routers    │
 * │                             │ EdgeDevice_3Mesh1Z_AsymmetricTopology    │ 3 routers    │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Connection Validation       │ ValidateConnection_VCAssignments         │ VCs          │
 * │                             │ ValidateConnection_DirectionMapping      │ Directions   │
 * │                             │ ValidateConnection_ChannelConsistency    │ Consistency  │
 * └─────────────────────────────┴──────────────────────────────────────────┴──────────────┘
 *
 * Total: 20 tests across 6 categories
 */
class RouterConnectionsTest : public ::testing::Test {
protected:
    void SetUp() override {
        registry_ = std::make_shared<ConnectionRegistry>();
    }

    void TearDown() override {
        registry_.reset();
    }

    std::shared_ptr<ConnectionRegistry> registry_;
};

// ============ Helper Functions ============

// Simulate recording a connection between two routers
void record_test_connection(
    const std::shared_ptr<ConnectionRegistry> &registry,
    FabricNodeId source_node,
    RoutingDirection source_dir,
    uint8_t source_eth_chan,
    uint32_t source_vc,
    uint32_t source_sender_ch,
    FabricNodeId dest_node,
    RoutingDirection dest_dir,
    uint8_t dest_eth_chan,
    uint32_t dest_vc,
    uint32_t dest_receiver_ch,
    ConnectionType conn_type) {
    RouterConnectionRecord record{
        .source_node = source_node,
        .source_direction = source_dir,
        .source_eth_chan = source_eth_chan,
        .source_vc = source_vc,
        .source_receiver_channel = source_sender_ch,
        .dest_node = dest_node,
        .dest_direction = dest_dir,
        .dest_eth_chan = dest_eth_chan,
        .dest_vc = dest_vc,
        .dest_sender_channel = dest_receiver_ch,
        .connection_type = conn_type
    };

    registry->record_connection(record);
}

// ============ Non-Z to Non-Z Router Tests (INTRA_MESH) ============

TEST_F(RouterConnectionsTest, MeshToMesh_VC0_Connection) {
    // Setup: Two mesh routers on different devices
    FabricNodeId device0(MeshId{0}, 0);
    FabricNodeId device1(MeshId{0}, 1);

    // Create channel mappings for both routers
    auto spec = MeshChannelSpec::create_for_compute_mesh(Topology::Mesh, nullptr);
    FabricRouterChannelMapping router0_mapping(
        Topology::Mesh,
        spec,
        false,  // no tensix
        RouterVariant::MESH);

    FabricRouterChannelMapping router1_mapping(
        Topology::Mesh,
        spec,
        false,  // no tensix
        RouterVariant::MESH);

    // Verify both routers have VC0 only
    EXPECT_EQ(spec.num_vcs, 1);
    EXPECT_EQ(spec.num_vcs, 1);

    // Simulate connection: router0 (EAST) → router1 (WEST), VC0 channel 1
    record_test_connection(
        registry_,
        device0, RoutingDirection::E, 0,  // source
        0, 1,  // VC0, sender channel 1
        device1, RoutingDirection::W, 0,  // dest
        0, 1,  // VC0, receiver channel (mapped to internal 1)
        ConnectionType::INTRA_MESH);

    // Verify connection was recorded
    EXPECT_EQ(registry_->size(), 1);

    auto connections = registry_->get_connections_by_type(ConnectionType::INTRA_MESH);
    ASSERT_EQ(connections.size(), 1);

    const auto& conn = connections[0];
    EXPECT_EQ(conn.source_node, device0);
    EXPECT_EQ(conn.source_direction, RoutingDirection::E);
    EXPECT_EQ(conn.source_vc, 0);
    EXPECT_EQ(conn.dest_node, device1);
    EXPECT_EQ(conn.dest_direction, RoutingDirection::W);
    EXPECT_EQ(conn.dest_vc, 0);
    EXPECT_EQ(conn.connection_type, ConnectionType::INTRA_MESH);
}

TEST_F(RouterConnectionsTest, MeshToMesh_Bidirectional_VC0) {
    // Setup: Two mesh routers with bidirectional connection
    FabricNodeId device0(MeshId{0}, 0);
    FabricNodeId device1(MeshId{0}, 1);

    // Connection 1: device0 → device1
    record_test_connection(
        registry_,
        device0, RoutingDirection::N, 0,
        0, 1,
        device1, RoutingDirection::S, 0,
        0, 1,
        ConnectionType::INTRA_MESH);

    // Connection 2: device1 → device0 (reverse)
    record_test_connection(
        registry_,
        device1, RoutingDirection::S, 0,
        0, 1,
        device0, RoutingDirection::N, 0,
        0, 1,
        ConnectionType::INTRA_MESH);

    EXPECT_EQ(registry_->size(), 2);

    // Verify both connections are INTRA_MESH
    auto intra_mesh = registry_->get_connections_by_type(ConnectionType::INTRA_MESH);
    EXPECT_EQ(intra_mesh.size(), 2);

    // Verify device0 has 1 outgoing connection
    auto device0_out = registry_->get_connections_from_source(device0, RoutingDirection::N);
    EXPECT_EQ(device0_out.size(), 1);

    // Verify device0 has 1 incoming connection
    auto device0_in = registry_->get_connections_to_dest(device0, RoutingDirection::N);
    EXPECT_EQ(device0_in.size(), 1);
}

TEST_F(RouterConnectionsTest, MeshToMesh_2D_MultipleChannels) {
    // Setup: 2D mesh router connecting to multiple neighbors
    FabricNodeId center(MeshId{0}, 4);  // Center of 3x3 grid
    FabricNodeId north(MeshId{0}, 1);
    FabricNodeId east(MeshId{0}, 5);
    FabricNodeId south(MeshId{0}, 7);
    FabricNodeId west(MeshId{0}, 3);

    // Center router connects to all 4 neighbors via VC0
    record_test_connection(registry_, center, RoutingDirection::N, 0, 0, 1,
                          north, RoutingDirection::S, 0, 0, 1, ConnectionType::INTRA_MESH);
    record_test_connection(registry_, center, RoutingDirection::E, 1, 0, 1,
                          east, RoutingDirection::W, 1, 0, 1, ConnectionType::INTRA_MESH);
    record_test_connection(registry_, center, RoutingDirection::S, 2, 0, 1,
                          south, RoutingDirection::N, 2, 0, 1, ConnectionType::INTRA_MESH);
    record_test_connection(registry_, center, RoutingDirection::W, 3, 0, 1,
                          west, RoutingDirection::E, 3, 0, 1, ConnectionType::INTRA_MESH);

    EXPECT_EQ(registry_->size(), 4);

    // Verify all are INTRA_MESH
    auto intra_mesh = registry_->get_connections_by_type(ConnectionType::INTRA_MESH);
    EXPECT_EQ(intra_mesh.size(), 4);
}

// ============ Non-Z to Z Router Tests (MESH_TO_Z) ============

TEST_F(RouterConnectionsTest, MeshToZ_VC0_Connection) {
    // Setup: Mesh router connecting to Z router on same device
    auto intermesh_config = IntermeshVCConfig::full_mesh();
    FabricNodeId device0(MeshId{0}, 0);

    // Mesh router (North direction)
    auto mesh_spec = MeshChannelSpec::create_for_compute_mesh(Topology::Mesh, nullptr);
    FabricRouterChannelMapping mesh_mapping(Topology::Mesh, mesh_spec, false, RouterVariant::MESH);

    // Z router
    auto z_spec = MeshChannelSpec::create_for_compute_mesh(Topology::Mesh, &intermesh_config);
    FabricRouterChannelMapping z_mapping(Topology::Mesh, z_spec, false, RouterVariant::Z_ROUTER);

    // Verify Z router has 2 VCs
    EXPECT_EQ(z_spec.num_vcs, 2);
    EXPECT_EQ(z_spec.sender_channels_per_vc[1], 4);

    // Simulate connection: mesh (VC0, sender ch 2) → Z (VC1, receiver ch 0)
    // Mesh router uses VC0 sender channel 2 to send to Z router
    // Z router receives on VC1 receiver channel 0 (mapped to erisc receiver 1)
    record_test_connection(
        registry_,
        device0, RoutingDirection::N, 0,  // mesh router
        0, 2,  // VC0, sender channel 2
        device0, RoutingDirection::Z, 4,  // Z router (eth_chan 4)
        1, 0,  // VC1, receiver channel 0
        ConnectionType::MESH_TO_Z);

    EXPECT_EQ(registry_->size(), 1);

    auto mesh_to_z = registry_->get_connections_by_type(ConnectionType::MESH_TO_Z);
    ASSERT_EQ(mesh_to_z.size(), 1);

    const auto& conn = mesh_to_z[0];
    EXPECT_EQ(conn.source_direction, RoutingDirection::N);
    EXPECT_EQ(conn.source_vc, 0);
    EXPECT_EQ(conn.dest_direction, RoutingDirection::Z);
    EXPECT_EQ(conn.dest_vc, 1);
    EXPECT_EQ(conn.connection_type, ConnectionType::MESH_TO_Z);

    // Verify Z router's VC1 receiver channel 0 maps to erisc receiver 1
    auto z_receiver_mapping = z_mapping.get_receiver_mapping(1, 0);
    EXPECT_EQ(z_receiver_mapping.builder_type, BuilderType::ERISC);
    EXPECT_EQ(z_receiver_mapping.internal_receiver_channel_id, 1);
}

TEST_F(RouterConnectionsTest, MeshToZ_MultipleMeshRouters) {
    // Setup: 4 mesh routers connecting to 1 Z router (typical device layout)
    FabricNodeId device0(MeshId{0}, 0);

    std::vector<RoutingDirection> mesh_dirs = {
        RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W
    };

    // Each mesh router connects to Z router via VC0 sender channel 2
    for (size_t i = 0; i < 4; ++i) {
        record_test_connection(
            registry_,
            device0, mesh_dirs[i], static_cast<uint8_t>(i),  // mesh router
            0, 2,  // VC0, sender channel 2
            device0, RoutingDirection::Z, 4,  // Z router
            1, 0,  // VC1, receiver channel 0 (same for all - multi-target)
            ConnectionType::MESH_TO_Z);
    }

    EXPECT_EQ(registry_->size(), 4);

    auto mesh_to_z = registry_->get_connections_by_type(ConnectionType::MESH_TO_Z);
    EXPECT_EQ(mesh_to_z.size(), 4);

    // Verify Z router receives from all 4 mesh routers
    auto z_incoming = registry_->get_connections_to_dest(device0, RoutingDirection::Z);
    EXPECT_EQ(z_incoming.size(), 4);

    // All should target VC1 receiver channel 0 (multi-target receiver)
    for (const auto& conn : z_incoming) {
        EXPECT_EQ(conn.dest_vc, 1);
        EXPECT_EQ(conn.dest_sender_channel, 0);  // All target same receiver
        EXPECT_EQ(conn.connection_type, ConnectionType::MESH_TO_Z);
    }

    // Verify Z router VC1 receiver can handle multiple sources
    // This is the multi-target receiver scenario
    std::set<RoutingDirection> source_dirs;
    for (const auto& conn : z_incoming) {
        source_dirs.insert(conn.source_direction);
    }
    EXPECT_EQ(source_dirs.size(), 4);  // 4 different source directions
}

// ============ Z to Non-Z Router Tests (Z_TO_MESH) ============

TEST_F(RouterConnectionsTest, ZToMesh_VC1_Connection) {
    // Setup: Z router connecting to mesh router on same device
    auto intermesh_config = IntermeshVCConfig::full_mesh();
    FabricNodeId device0(MeshId{0}, 0);

    // Z router
    auto z_spec = MeshChannelSpec::create_for_compute_mesh(Topology::Mesh, &intermesh_config);
    FabricRouterChannelMapping z_mapping(Topology::Mesh, z_spec, false, RouterVariant::Z_ROUTER);

    // Mesh router
    auto mesh_spec = MeshChannelSpec::create_for_compute_mesh(Topology::Mesh, nullptr);
    FabricRouterChannelMapping mesh_mapping(Topology::Mesh, mesh_spec, false, RouterVariant::MESH);

    // Verify Z router VC1 has 4 sender channels
    EXPECT_EQ(z_spec.sender_channels_per_vc[1], 4);

    // Verify Z router VC1 sender channels map to erisc 4-7
    for (uint32_t i = 0; i < 4; ++i) {
        auto sender_mapping = z_mapping.get_sender_mapping(1, i);
        EXPECT_EQ(sender_mapping.builder_type, BuilderType::ERISC);
        EXPECT_EQ(sender_mapping.internal_sender_channel_id, 4 + i);
    }

    // Simulate connection: Z (VC1, sender ch 0) → mesh North (VC1, receiver)
    record_test_connection(
        registry_,
        device0, RoutingDirection::Z, 4,  // Z router
        1, 0,  // VC1, sender channel 0 (maps to erisc 4)
        device0, RoutingDirection::N, 0,  // mesh router
        1, 0,  // VC1, receiver channel 0
        ConnectionType::Z_TO_MESH);

    EXPECT_EQ(registry_->size(), 1);

    auto z_to_mesh = registry_->get_connections_by_type(ConnectionType::Z_TO_MESH);
    ASSERT_EQ(z_to_mesh.size(), 1);

    const auto& conn = z_to_mesh[0];
    EXPECT_EQ(conn.source_direction, RoutingDirection::Z);
    EXPECT_EQ(conn.source_vc, 1);
    EXPECT_EQ(conn.source_receiver_channel, 0);
    EXPECT_EQ(conn.dest_direction, RoutingDirection::N);
    EXPECT_EQ(conn.dest_vc, 1);
    EXPECT_EQ(conn.connection_type, ConnectionType::Z_TO_MESH);
}

TEST_F(RouterConnectionsTest, ZToMesh_AllFourDirections) {
    // Setup: Z router connecting to all 4 mesh routers
    FabricNodeId device0(MeshId{0}, 0);

    std::vector<RoutingDirection> mesh_dirs = {
        RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W
    };

    // Z router uses VC1 sender channels 0-3 to connect to 4 mesh routers VC1
    for (size_t i = 0; i < 4; ++i) {
        record_test_connection(
            registry_,
            device0, RoutingDirection::Z, 4,  // Z router
            1, static_cast<uint32_t>(i),  // VC1, sender channel i
            device0, mesh_dirs[i], static_cast<uint8_t>(i),  // mesh router
            1, 0,  // VC1, receiver channel 0
            ConnectionType::Z_TO_MESH);
    }

    EXPECT_EQ(registry_->size(), 4);

    auto z_to_mesh = registry_->get_connections_by_type(ConnectionType::Z_TO_MESH);
    EXPECT_EQ(z_to_mesh.size(), 4);

    // Verify Z router has 4 outgoing connections
    auto z_outgoing = registry_->get_connections_from_source(device0, RoutingDirection::Z);
    EXPECT_EQ(z_outgoing.size(), 4);

    // Each should use a different sender channel (0-3)
    std::set<uint32_t> sender_channels;
    for (const auto& conn : z_outgoing) {
        EXPECT_EQ(conn.source_vc, 1);
        sender_channels.insert(conn.source_receiver_channel);
    }
    EXPECT_EQ(sender_channels.size(), 4);  // All 4 channels used
}

TEST_F(RouterConnectionsTest, ZToMesh_VariableRouterCount) {
    // Setup: Z router connecting to only 2 mesh routers (edge device scenario)
    FabricNodeId device0(MeshId{0}, 0);

    // Only North and East mesh routers present
    record_test_connection(
        registry_,
        device0, RoutingDirection::Z, 4,
        1, 0,  // VC1, sender channel 0
        device0, RoutingDirection::N, 0,
        1, 0,  // VC1, receiver channel 0
        ConnectionType::Z_TO_MESH);

    record_test_connection(
        registry_,
        device0, RoutingDirection::Z, 4,
        1, 1,  // VC1, sender channel 1
        device0, RoutingDirection::E, 1,
        1, 0,  // VC1, receiver channel 0
        ConnectionType::Z_TO_MESH);

    EXPECT_EQ(registry_->size(), 2);

    auto z_to_mesh = registry_->get_connections_by_type(ConnectionType::Z_TO_MESH);
    EXPECT_EQ(z_to_mesh.size(), 2);

    // Verify only 2 connections from Z router
    auto z_outgoing = registry_->get_connections_from_source(device0, RoutingDirection::Z);
    EXPECT_EQ(z_outgoing.size(), 2);
}

// ============ Complete Device Scenario ============

TEST_F(RouterConnectionsTest, CompleteDevice_WithZRouter) {
    // Setup: Complete device with 4 mesh routers and 1 Z router
    // - Mesh routers connect to each other (INTRA_MESH)
    // - Mesh routers connect to Z router (MESH_TO_Z)
    // - Z router connects to mesh routers (Z_TO_MESH)

    FabricNodeId device0(MeshId{0}, 0);
    FabricNodeId device1(MeshId{0}, 1);  // External device

    // 1. INTRA_MESH: device0 North router → device1 South router
    record_test_connection(
        registry_,
        device0, RoutingDirection::N, 0,
        0, 1,
        device1, RoutingDirection::S, 0,
        0, 1,
        ConnectionType::INTRA_MESH);

    // 2. MESH_TO_Z: device0 North router → device0 Z router
    record_test_connection(
        registry_,
        device0, RoutingDirection::N, 0,
        0, 2,
        device0, RoutingDirection::Z, 4,
        1, 0,
        ConnectionType::MESH_TO_Z);

    // 3. Z_TO_MESH: device0 Z router VC1 → device0 North router VC1
    record_test_connection(
        registry_,
        device0, RoutingDirection::Z, 4,
        1, 0,
        device0, RoutingDirection::N, 0,
        1, 0,
        ConnectionType::Z_TO_MESH);

    EXPECT_EQ(registry_->size(), 3);

    // Verify connection types
    EXPECT_EQ(registry_->get_connections_by_type(ConnectionType::INTRA_MESH).size(), 1);
    EXPECT_EQ(registry_->get_connections_by_type(ConnectionType::MESH_TO_Z).size(), 1);
    EXPECT_EQ(registry_->get_connections_by_type(ConnectionType::Z_TO_MESH).size(), 1);

    // Verify North router has 2 outgoing connections (INTRA_MESH + MESH_TO_Z)
    auto north_out = registry_->get_connections_from_source(device0, RoutingDirection::N);
    EXPECT_EQ(north_out.size(), 2);

    // Verify Z router has 1 outgoing and 1 incoming
    auto z_out = registry_->get_connections_from_source(device0, RoutingDirection::Z);
    auto z_in = registry_->get_connections_to_dest(device0, RoutingDirection::Z);
    EXPECT_EQ(z_out.size(), 1);
    EXPECT_EQ(z_in.size(), 1);
}

TEST_F(RouterConnectionsTest, FullMesh_4Routers_WithZ) {
    // Setup: Complete device topology
    // 4 mesh routers (N, E, S, W) + 1 Z router
    // Each mesh connects to 2 adjacent mesh routers + Z router
    // Z router connects to all 4 mesh routers

    FabricNodeId device0(MeshId{0}, 0);

    // INTRA_MESH connections (simplified - just a few)
    record_test_connection(registry_, device0, RoutingDirection::N, 0, 0, 1,
                          device0, RoutingDirection::E, 1, 0, 1, ConnectionType::INTRA_MESH);
    record_test_connection(registry_, device0, RoutingDirection::E, 1, 0, 1,
                          device0, RoutingDirection::S, 2, 0, 1, ConnectionType::INTRA_MESH);

    // MESH_TO_Z connections (all 4 mesh routers → Z)
    std::vector<RoutingDirection> dirs = {
        RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W
    };
    for (size_t i = 0; i < 4; ++i) {
        record_test_connection(
            registry_,
            device0, dirs[i], static_cast<uint8_t>(i),
            0, 2,
            device0, RoutingDirection::Z, 4,
            1, 0,
            ConnectionType::MESH_TO_Z);
    }

    // Z_TO_MESH connections (Z VC1 → all 4 mesh routers VC1)
    for (size_t i = 0; i < 4; ++i) {
        record_test_connection(
            registry_,
            device0, RoutingDirection::Z, 4,
            1, static_cast<uint32_t>(i),
            device0, dirs[i], static_cast<uint8_t>(i),
            1, 0,
            ConnectionType::Z_TO_MESH);
    }

    // Total: 2 INTRA_MESH + 4 MESH_TO_Z + 4 Z_TO_MESH = 10 connections
    EXPECT_EQ(registry_->size(), 10);
    EXPECT_EQ(registry_->get_connections_by_type(ConnectionType::INTRA_MESH).size(), 2);
    EXPECT_EQ(registry_->get_connections_by_type(ConnectionType::MESH_TO_Z).size(), 4);
    EXPECT_EQ(registry_->get_connections_by_type(ConnectionType::Z_TO_MESH).size(), 4);

    // Verify Z router connectivity
    auto z_out = registry_->get_connections_from_source(device0, RoutingDirection::Z);
    auto z_in = registry_->get_connections_to_dest(device0, RoutingDirection::Z);
    EXPECT_EQ(z_out.size(), 4);  // Z → 4 mesh routers
    EXPECT_EQ(z_in.size(), 4);   // 4 mesh routers → Z
}

// ============ Multi-Target Receiver Validation ============

TEST_F(RouterConnectionsTest, Phase1_5_ZRouter_MultiTargetReceiver_Validation) {
    // This test validates the multi-target receiver scenario:
    // Z router VC1 receiver channel 0 receives from 4 different mesh routers
    // This was previously impossible due to fixed array overwriting

    auto intermesh_config = IntermeshVCConfig::full_mesh();
    FabricNodeId device0(MeshId{0}, 0);

    // Z router channel mapping
    auto z_spec = MeshChannelSpec::create_for_compute_mesh(Topology::Mesh, &intermesh_config);
    FabricRouterChannelMapping z_mapping(Topology::Mesh, z_spec, false, RouterVariant::Z_ROUTER);

    // Verify Z router has correct VC1 layout
    EXPECT_EQ(z_spec.num_vcs, 2);
    EXPECT_EQ(z_spec.sender_channels_per_vc[1], 4);

    // Verify VC1 receiver channel 0 maps to erisc receiver 1
    auto vc1_receiver = z_mapping.get_receiver_mapping(1, 0);
    EXPECT_EQ(vc1_receiver.builder_type, BuilderType::ERISC);
    EXPECT_EQ(vc1_receiver.internal_receiver_channel_id, 1);

    // Record 4 MESH_TO_Z connections, all targeting same receiver
    std::vector<RoutingDirection> mesh_dirs = {
        RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W
    };

    for (size_t i = 0; i < 4; ++i) {
        // Create mesh router mapping
        auto mesh_spec = MeshChannelSpec::create_for_compute_mesh(Topology::Mesh, nullptr);
        FabricRouterChannelMapping mesh_mapping(Topology::Mesh, mesh_spec, false, RouterVariant::MESH);

        // Verify mesh router VC0 sender channel 2 exists
        EXPECT_GE(mesh_spec.sender_channels_per_vc[0], 3);

        // Record connection: mesh VC0 ch2 → Z VC1 ch0
        record_test_connection(
            registry_,
            device0, mesh_dirs[i], static_cast<uint8_t>(i),
            0, 2,  // VC0, sender channel 2
            device0, RoutingDirection::Z, 4,
            1, 0,  // VC1, receiver channel 0 (SAME for all 4)
            ConnectionType::MESH_TO_Z);
    }

    // Validate multi-target receiver scenario
    auto z_incoming = registry_->get_connections_to_dest(device0, RoutingDirection::Z);
    ASSERT_EQ(z_incoming.size(), 4);

    // Critical validation: All 4 connections target the SAME receiver
    for (const auto& conn : z_incoming) {
        EXPECT_EQ(conn.dest_vc, 1);
        EXPECT_EQ(conn.dest_sender_channel, 0);  // Same receiver for all!
        EXPECT_EQ(conn.dest_direction, RoutingDirection::Z);
    }

    // Verify all 4 source directions are present
    std::set<RoutingDirection> sources;
    for (const auto& conn : z_incoming) {
        sources.insert(conn.source_direction);
    }
    EXPECT_EQ(sources.size(), 4);
    EXPECT_TRUE(sources.count(RoutingDirection::N) > 0);
    EXPECT_TRUE(sources.count(RoutingDirection::E) > 0);
    EXPECT_TRUE(sources.count(RoutingDirection::S) > 0);
    EXPECT_TRUE(sources.count(RoutingDirection::W) > 0);
}

TEST_F(RouterConnectionsTest, Phase1_5_EdgeDevice_VariableTargetCount) {
    // Multi-target receiver supports variable target counts (2-4 mesh routers)
    // This test validates an edge device with only 2 mesh routers

    FabricNodeId device0(MeshId{0}, 0);

    // Only North and East mesh routers present (edge device)
    record_test_connection(
        registry_,
        device0, RoutingDirection::N, 0,
        0, 2,
        device0, RoutingDirection::Z, 4,
        1, 0,  // Same receiver channel
        ConnectionType::MESH_TO_Z);

    record_test_connection(
        registry_,
        device0, RoutingDirection::E, 1,
        0, 2,
        device0, RoutingDirection::Z, 4,
        1, 0,  // Same receiver channel
        ConnectionType::MESH_TO_Z);

    auto z_incoming = registry_->get_connections_to_dest(device0, RoutingDirection::Z);
    EXPECT_EQ(z_incoming.size(), 2);

    // Both target same receiver (multi-target with count=2)
    for (const auto& conn : z_incoming) {
        EXPECT_EQ(conn.dest_sender_channel, 0);
    }
}

TEST_F(RouterConnectionsTest, Phase1_5_Bidirectional_ZAndMesh) {
    // Comprehensive test: Z router both receives from and sends to mesh routers
    // This validates both MESH_TO_Z (multi-target receiver) and Z_TO_MESH

    FabricNodeId device0(MeshId{0}, 0);

    // 4 mesh routers
    std::vector<RoutingDirection> mesh_dirs = {
        RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W
    };

    // MESH_TO_Z: All 4 mesh routers → Z router VC1 receiver 0 (multi-target)
    for (size_t i = 0; i < 4; ++i) {
        record_test_connection(
            registry_,
            device0, mesh_dirs[i], static_cast<uint8_t>(i),
            0, 2,  // mesh VC0 sender 2
            device0, RoutingDirection::Z, 4,
            1, 0,  // Z VC1 receiver 0 (SAME for all)
            ConnectionType::MESH_TO_Z);
    }

    // Z_TO_MESH: Z router VC1 senders 0-3 → 4 mesh routers (one-to-one)
    for (size_t i = 0; i < 4; ++i) {
        record_test_connection(
            registry_,
            device0, RoutingDirection::Z, 4,
            1, static_cast<uint32_t>(i),  // Z VC1 sender i
            device0, mesh_dirs[i], static_cast<uint8_t>(i),
            1, 0,  // mesh VC1 receiver
            ConnectionType::Z_TO_MESH);
    }

    // Validate counts
    EXPECT_EQ(registry_->size(), 8);
    EXPECT_EQ(registry_->get_connections_by_type(ConnectionType::MESH_TO_Z).size(), 4);
    EXPECT_EQ(registry_->get_connections_by_type(ConnectionType::Z_TO_MESH).size(), 4);

    // Validate Z router connectivity
    auto z_in = registry_->get_connections_to_dest(device0, RoutingDirection::Z);
    auto z_out = registry_->get_connections_from_source(device0, RoutingDirection::Z);

    EXPECT_EQ(z_in.size(), 4);   // 4 incoming (multi-target receiver)
    EXPECT_EQ(z_out.size(), 4);  // 4 outgoing (one per sender channel)

    // All incoming target same receiver (multi-target)
    for (const auto& conn : z_in) {
        EXPECT_EQ(conn.dest_sender_channel, 0);
    }

    // All outgoing use different sender channels
    std::set<uint32_t> sender_channels;
    for (const auto& conn : z_out) {
        sender_channels.insert(conn.source_receiver_channel);
    }
    EXPECT_EQ(sender_channels.size(), 4);  // Channels 0, 1, 2, 3
}

// ============ Mapping-Driven Connection Tests ============

TEST_F(RouterConnectionsTest, Phase2_MappingDriven_MeshToMesh_1D) {
    // Use connection mapping to drive connection setup

    // Create mappings for two 1D mesh routers
    auto north_mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Linear,
        RoutingDirection::N,
        false);

    auto south_mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Linear,
        RoutingDirection::S,
        false);

    FabricNodeId north_node(MeshId{0}, 0);
    FabricNodeId south_node(MeshId{0}, 1);

    // North router connects to South (driven by mapping)
    auto north_targets = north_mapping.get_downstream_targets(0, 0);
    ASSERT_EQ(north_targets.size(), 1);
    EXPECT_EQ(north_targets[0].target_direction.value(), RoutingDirection::S);

    record_test_connection(
        registry_,
        north_node, RoutingDirection::N, 0,
        0, 0,  // VC0, receiver channel 0
        south_node, RoutingDirection::S, 0,
        0, north_targets[0].target_sender_channel,
        ConnectionType::INTRA_MESH);

    // South router connects to North (driven by mapping)
    auto south_targets = south_mapping.get_downstream_targets(0, 0);
    ASSERT_EQ(south_targets.size(), 1);
    EXPECT_EQ(south_targets[0].target_direction.value(), RoutingDirection::N);

    record_test_connection(
        registry_,
        south_node, RoutingDirection::S, 0,
        0, 0,  // VC0, receiver channel 0
        north_node, RoutingDirection::N, 0,
        0, south_targets[0].target_sender_channel,
        ConnectionType::INTRA_MESH);

    // Verify bidirectional connection
    EXPECT_EQ(registry_->size(), 2);

    auto north_out = registry_->get_connections_by_source_node(north_node);
    auto south_out = registry_->get_connections_by_source_node(south_node);

    EXPECT_EQ(north_out.size(), 1);
    EXPECT_EQ(south_out.size(), 1);
}

TEST_F(RouterConnectionsTest, Phase2_MappingDriven_MeshToMesh_2D) {
    // Use connection mapping for 2D mesh

    // Create mapping for NORTH router in 2D mesh
    auto north_mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh,
        RoutingDirection::N,
        false);

    FabricNodeId north_node(MeshId{0}, 0);

    // NORTH router receiver channel 0 should have 3 targets for INTRA_MESH
    EXPECT_EQ(north_mapping.get_total_sender_count(), 3);

    // Get all targets from receiver channel 0
    auto targets = north_mapping.get_downstream_targets(0, 0);
    ASSERT_EQ(targets.size(), 3);

    // Verify one target → SOUTH (opposite)
    auto south_it = std::find_if(targets.begin(), targets.end(),
        [](const ConnectionTarget& t) { return t.target_direction == RoutingDirection::S; });
    ASSERT_NE(south_it, targets.end());
    EXPECT_EQ(south_it->type, ConnectionType::INTRA_MESH);

    // Verify cross directions (EAST/WEST) are present
    std::set<RoutingDirection> all_dirs;
    for (const auto& target : targets) {
        all_dirs.insert(target.target_direction.value());
    }

    EXPECT_TRUE(all_dirs.count(RoutingDirection::S));
    EXPECT_TRUE(all_dirs.count(RoutingDirection::E));
    EXPECT_TRUE(all_dirs.count(RoutingDirection::W));

    // Record all connections
    FabricNodeId south_node(MeshId{0}, 1);
    FabricNodeId east_node(MeshId{0}, 2);
    FabricNodeId west_node(MeshId{0}, 3);

    record_test_connection(
        registry_,
        north_node, RoutingDirection::N, 0,
        0, 1,
        south_node, RoutingDirection::S, 0,
        0, 0,
        ConnectionType::INTRA_MESH);

    record_test_connection(
        registry_,
        north_node, RoutingDirection::N, 0,
        0, 2,
        east_node, RoutingDirection::E, 0,
        0, 0,
        ConnectionType::INTRA_MESH);

    record_test_connection(
        registry_,
        north_node, RoutingDirection::N, 0,
        0, 3,
        west_node, RoutingDirection::W, 0,
        0, 0,
        ConnectionType::INTRA_MESH);

    EXPECT_EQ(registry_->size(), 3);
}

TEST_F(RouterConnectionsTest, Phase2_MappingDriven_MeshToZ) {
    // Use connection mapping to set up MESH_TO_Z connection

    // Create mapping for mesh router with Z
    auto mesh_mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh,
        RoutingDirection::N,
        true);  // Has Z router

    FabricNodeId mesh_node(MeshId{0}, 0);
    FabricNodeId z_node(MeshId{0}, 100);

    // Verify mapping has 4 sender channels (3 INTRA_MESH + 1 MESH_TO_Z)
    EXPECT_EQ(mesh_mapping.get_total_sender_count(), 4);

    // Receiver channel 0 should have MESH_TO_Z target
    auto all_targets = mesh_mapping.get_downstream_targets(0, 0);
    auto z_target_it = std::find_if(all_targets.begin(), all_targets.end(),
        [](const ConnectionTarget& t) { return t.type == ConnectionType::MESH_TO_Z; });
    ASSERT_NE(z_target_it, all_targets.end());
    EXPECT_EQ(z_target_it->target_direction.value(), RoutingDirection::Z);

    // Record MESH_TO_Z connection
    record_test_connection(
        registry_,
        mesh_node, RoutingDirection::N, 0,
        0, 0,  // VC0, receiver channel 0
        z_node, RoutingDirection::Z, 0,
        0, z_target_it->target_sender_channel,
        ConnectionType::MESH_TO_Z);

    EXPECT_EQ(registry_->size(), 1);

    auto mesh_to_z = registry_->get_connections_by_type(ConnectionType::MESH_TO_Z);
    EXPECT_EQ(mesh_to_z.size(), 1);
    EXPECT_EQ(mesh_to_z[0].source_receiver_channel, 0);
}

TEST_F(RouterConnectionsTest, Phase2_MappingDriven_ZToMesh_AllDirections) {
    // Use connection mapping to set up Z_TO_MESH connections

    // Create mapping for Z router
    auto z_mapping = RouterConnectionMapping::for_z_router();

    FabricNodeId z_node(MeshId{0}, 100);

    // Z router should have 4 sender channels on VC1
    EXPECT_EQ(z_mapping.get_total_sender_count(), 4);

    // Get all targets from receiver channel 0 on VC1
    auto targets = z_mapping.get_downstream_targets(1, 0);
    ASSERT_EQ(targets.size(), 4);

    // Verify each target maps to correct direction
    std::vector<RoutingDirection> expected_dirs = {
        RoutingDirection::N,
        RoutingDirection::E,
        RoutingDirection::S,
        RoutingDirection::W
    };

    for (size_t i = 0; i < targets.size(); ++i) {
        EXPECT_EQ(targets[i].type, ConnectionType::Z_TO_MESH);
        EXPECT_EQ(targets[i].target_vc, 1);  // Target mesh router VC1

        // Record connection
        FabricNodeId mesh_node(MeshId{0}, i);
        record_test_connection(
            registry_,
            z_node, RoutingDirection::Z, 0,
            1, 0,  // VC1, receiver channel 0
            mesh_node, targets[i].target_direction.value(), 0,
            1, targets[i].target_sender_channel,
            ConnectionType::Z_TO_MESH);
    }

    EXPECT_EQ(registry_->size(), 4);

    auto z_to_mesh = registry_->get_connections_by_type(ConnectionType::Z_TO_MESH);
    EXPECT_EQ(z_to_mesh.size(), 4);

    // Verify all use VC1 (Z VC1 → mesh VC1)
    for (const auto& conn : z_to_mesh) {
        EXPECT_EQ(conn.source_vc, 1);
        EXPECT_EQ(conn.dest_vc, 1);
    }
}

TEST_F(RouterConnectionsTest, Phase2_MappingDriven_FullDevice_Connections) {
    // Simulate FabricBuilder connection establishment using connection mappings
    // Full device: 4 mesh routers + 1 Z router

    std::vector<RoutingDirection> mesh_directions = {
        RoutingDirection::N,
        RoutingDirection::E,
        RoutingDirection::S,
        RoutingDirection::W
    };

    FabricNodeId z_node(MeshId{0}, 100);

    // Step 1: Create mesh router mappings and record MESH_TO_Z connections
    for (size_t i = 0; i < 4; ++i) {
        auto mesh_mapping = RouterConnectionMapping::for_mesh_router(
            Topology::Mesh,
            mesh_directions[i],
            true);  // Has Z router

        FabricNodeId mesh_node(MeshId{0}, i);

        // Get MESH_TO_Z target from receiver channel 0
        auto all_targets = mesh_mapping.get_downstream_targets(0, 0);
        auto z_target_it = std::find_if(all_targets.begin(), all_targets.end(),
            [](const ConnectionTarget& t) { return t.type == ConnectionType::MESH_TO_Z; });
        ASSERT_NE(z_target_it, all_targets.end());

        record_test_connection(
            registry_,
            mesh_node, mesh_directions[i], static_cast<uint8_t>(i),
            0, 0,  // VC0, receiver channel 0
            z_node, RoutingDirection::Z, 0,
            0, 0,
            ConnectionType::MESH_TO_Z);
    }

    // Step 2: Create Z router mapping and record Z_TO_MESH connections
    auto z_mapping = RouterConnectionMapping::for_z_router();

    // Get all targets from receiver channel 0 on VC1
    auto z_targets = z_mapping.get_downstream_targets(1, 0);
    ASSERT_EQ(z_targets.size(), 4);

    for (size_t i = 0; i < z_targets.size(); ++i) {
        FabricNodeId mesh_node(MeshId{0}, i);

        record_test_connection(
            registry_,
            z_node, RoutingDirection::Z, 0,
            1, 0,  // VC1, receiver channel 0
            mesh_node, mesh_directions[i], static_cast<uint8_t>(i),
            1, z_targets[i].target_sender_channel,
            ConnectionType::Z_TO_MESH);
    }

    // Verify total connections: 4 MESH_TO_Z + 4 Z_TO_MESH = 8
    EXPECT_EQ(registry_->size(), 8);

    auto mesh_to_z = registry_->get_connections_by_type(ConnectionType::MESH_TO_Z);
    auto z_to_mesh = registry_->get_connections_by_type(ConnectionType::Z_TO_MESH);

    EXPECT_EQ(mesh_to_z.size(), 4);
    EXPECT_EQ(z_to_mesh.size(), 4);

    // Verify Z router receives from all 4 mesh routers
    auto z_incoming = registry_->get_connections_by_dest_node(z_node);
    EXPECT_EQ(z_incoming.size(), 4);

    // Verify Z router sends to all 4 mesh routers
    auto z_outgoing = registry_->get_connections_by_source_node(z_node);
    EXPECT_EQ(z_outgoing.size(), 4);
}

TEST_F(RouterConnectionsTest, Phase2_MappingDriven_EdgeDevice_DynamicSizing) {
    // Edge device with only 2 mesh routers (NORTH and EAST) + Z router
    // Demonstrates dynamic sizing: Z mapping has 4 intents, only 2 realized

    std::vector<RoutingDirection> existing_mesh = {
        RoutingDirection::N,
        RoutingDirection::E
    };

    FabricNodeId z_node(MeshId{0}, 100);

    // Step 1: Mesh routers connect to Z
    for (size_t i = 0; i < 2; ++i) {
        auto mesh_mapping = RouterConnectionMapping::for_mesh_router(
            Topology::Mesh,
            existing_mesh[i],
            true);

        FabricNodeId mesh_node(MeshId{0}, i);

        auto all_targets = mesh_mapping.get_downstream_targets(0, 0);
        auto z_target_it = std::find_if(all_targets.begin(), all_targets.end(),
            [](const ConnectionTarget& t) { return t.type == ConnectionType::MESH_TO_Z; });
        ASSERT_NE(z_target_it, all_targets.end());

        record_test_connection(
            registry_,
            mesh_node, existing_mesh[i], static_cast<uint8_t>(i),
            0, 0,  // VC0, receiver channel 0
            z_node, RoutingDirection::Z, 0,
            0, 0,
            ConnectionType::MESH_TO_Z);
    }

    // Step 2: Z router connects to mesh (only existing ones)
    auto z_mapping = RouterConnectionMapping::for_z_router();

    // Z mapping has 4 intents, but we only realize 2
    std::set<RoutingDirection> existing_set(existing_mesh.begin(), existing_mesh.end());

    auto z_targets = z_mapping.get_downstream_targets(1, 0);
    ASSERT_EQ(z_targets.size(), 4);

    // Only record connections for existing routers
    for (const auto& target : z_targets) {
        if (existing_set.count(target.target_direction.value())) {
            FabricNodeId mesh_node(MeshId{0}, 0);  // Simplified for test

            record_test_connection(
                registry_,
                z_node, RoutingDirection::Z, 0,
                1, 0,  // VC1, receiver channel 0
                mesh_node, target.target_direction.value(), 0,
                1, target.target_sender_channel,
                ConnectionType::Z_TO_MESH);
        }
    }

    // Verify connections: 2 MESH_TO_Z + 2 Z_TO_MESH = 4
    EXPECT_EQ(registry_->size(), 4);

    auto mesh_to_z = registry_->get_connections_by_type(ConnectionType::MESH_TO_Z);
    auto z_to_mesh = registry_->get_connections_by_type(ConnectionType::Z_TO_MESH);

    EXPECT_EQ(mesh_to_z.size(), 2);
    EXPECT_EQ(z_to_mesh.size(), 2);
}

TEST_F(RouterConnectionsTest, Phase2_ChannelMapping_And_ConnectionMapping_Consistency) {
    // Verify channel mapping and connection mapping are consistent

    // Create Z router channel mapping
    auto intermesh_config = IntermeshVCConfig::full_mesh();
    auto z_spec = MeshChannelSpec::create_for_compute_mesh(Topology::Mesh, &intermesh_config);
    FabricRouterChannelMapping z_channel_mapping(Topology::Mesh, z_spec, false, RouterVariant::Z_ROUTER);

    // Create Z router connection mapping
    auto z_conn_mapping = RouterConnectionMapping::for_z_router();

    // Verify VC1 sender channel count matches target count
    uint32_t channel_map_senders = z_spec.sender_channels_per_vc[1];
    uint32_t conn_map_targets = z_conn_mapping.get_total_sender_count();

    EXPECT_EQ(channel_map_senders, 4);
    EXPECT_EQ(conn_map_targets, 4);

    // Verify receiver channel 0 has targets matching sender count
    EXPECT_TRUE(z_conn_mapping.has_targets(1, 0))
        << "Receiver channel 0 should have connection targets";

    auto targets = z_conn_mapping.get_downstream_targets(1, 0);
    EXPECT_EQ(targets.size(), channel_map_senders)
        << "Receiver channel 0 should have " << channel_map_senders << " targets";
}

}  // namespace tt::tt_fabric
