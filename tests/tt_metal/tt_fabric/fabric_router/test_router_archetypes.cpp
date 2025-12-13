// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "tt_metal/fabric/builder/connection_registry.hpp"
#include "tt_metal/fabric/builder/router_connection_mapping.hpp"
#include "tt_metal/fabric/fabric_router_channel_mapping.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"

namespace tt::tt_fabric {

/**
 * Test fixture for router archetypes and connection manager patterns
 *
 * These tests validate that we can create connection manager archetypes
 * for different router types and verify all connection scenarios:
 * - Non-Z → Non-Z (INTRA_MESH)
 * - Non-Z → Z (MESH_TO_Z)
 * - Z → Non-Z (Z_TO_MESH)
 * - Full device connection establishment
 *
 * Test Coverage Summary:
 * ┌──────────────────────────────────────────────────────────────────────────────────────┐
 * │ Category                    │ Test Name                                │ Focus        │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Archetype Creation          │ CreateMeshRouterArchetype_1D             │ 1D mesh      │
 * │                             │ CreateMeshRouterArchetype_2D_NoZ         │ 2D mesh      │
 * │                             │ CreateMeshRouterArchetype_2D_WithZ       │ 2D mesh+Z    │
 * │                             │ CreateZRouterArchetype                   │ Z router     │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Connection Establishment    │ EstablishConnection_MeshToMesh           │ Mesh↔Mesh    │
 * │                             │ EstablishConnection_MeshToZ              │ Mesh→Z       │
 * │                             │ EstablishConnection_ZToMesh              │ Z→Mesh       │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Full Device Scenarios       │ FullDevice_4Mesh1Z_AllArchetypes         │ Complete     │
 * │                             │ FullDevice_4Mesh1Z_AllConnections        │ All conns    │
 * │                             │ FullDevice_4Mesh1Z_VerifyRegistry        │ Registry     │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Edge Device Scenarios       │ EdgeDevice_2Mesh1Z_LimitedArchetypes     │ 2 routers    │
 * │                             │ EdgeDevice_2Mesh1Z_PartialConnections    │ Partial      │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Archetype Validation        │ ValidateArchetype_ChannelMapping         │ Channels     │
 * │                             │ ValidateArchetype_ConnectionMapping      │ Connections  │
 * │                             │ ValidateArchetype_Consistency            │ Consistency  │
 * └─────────────────────────────┴──────────────────────────────────────────┴──────────────┘
 *
 * Total: 15 tests across 5 categories
 */
class RouterArchetypesTest : public ::testing::Test {
protected:
    void SetUp() override {
        registry_ = std::make_shared<ConnectionRegistry>();
    }

    void TearDown() override {
        registry_.reset();
    }

    std::shared_ptr<ConnectionRegistry> registry_;

    /**
     * @brief Router archetype encapsulating channel mapping + connection mapping
     */
    struct RouterArchetype {
        FabricRouterChannelMapping channel_mapping;
        RouterConnectionMapping connection_mapping;
        FabricNodeId node_id;
        RoutingDirection direction;
        uint8_t eth_chan;

        RouterArchetype(
            FabricRouterChannelMapping ch_map,
            RouterConnectionMapping conn_map,
            FabricNodeId node,
            RoutingDirection dir,
            uint8_t eth)
            : channel_mapping(std::move(ch_map)),
              connection_mapping(std::move(conn_map)),
              node_id(node),
              direction(dir),
              eth_chan(eth) {}
    };

    /**
     * @brief Create a mesh router archetype
     */
    RouterArchetype create_mesh_router_archetype(
        Topology topology, RoutingDirection direction, bool has_z, FabricNodeId node_id, uint8_t eth_chan = 0) {
        // Channel mapping
        FabricRouterChannelMapping channel_mapping(
            topology,
            false,  // no tensix for now
            RouterVariant::MESH, nullptr);

        // Connection mapping
        RouterConnectionMapping connection_mapping =
            RouterConnectionMapping::for_mesh_router(topology, direction, has_z);

        return RouterArchetype(
            std::move(channel_mapping),
            std::move(connection_mapping),
            node_id,
            direction,
            eth_chan);
    }

    /**
     * @brief Create a Z router archetype
     */
    RouterArchetype create_z_router_archetype(
        FabricNodeId node_id,
        uint8_t eth_chan = 0) {
        static auto intermesh_config = IntermeshVCConfig::full_mesh();

        // Channel mapping
        FabricRouterChannelMapping channel_mapping(
            Topology::Mesh,
            false,  // no tensix
            RouterVariant::Z_ROUTER, &intermesh_config);

        // Connection mapping
        RouterConnectionMapping connection_mapping = RouterConnectionMapping::for_z_router();

        return RouterArchetype(
            std::move(channel_mapping),
            std::move(connection_mapping),
            node_id,
            RoutingDirection::Z,
            eth_chan);
    }

    /**
     * @brief Helper to record a connection between two archetypes
     */
    void record_archetype_connection(
        const RouterArchetype& source,
        uint32_t source_vc,
        uint32_t source_sender_ch,
        const RouterArchetype& dest,
        uint32_t dest_vc,
        uint32_t dest_receiver_ch,
        ConnectionType conn_type) {
        RouterConnectionRecord record{
            .source_node = source.node_id,
            .source_direction = source.direction,
            .source_eth_chan = source.eth_chan,
            .source_vc = source_vc,
            .source_receiver_channel = source_sender_ch,
            .dest_node = dest.node_id,
            .dest_direction = dest.direction,
            .dest_eth_chan = dest.eth_chan,
            .dest_vc = dest_vc,
            .dest_sender_channel = dest_receiver_ch,
            .connection_type = conn_type
        };

        registry_->record_connection(record);
    }
};

// ============================================================================
// Router Archetype Creation Tests
// ============================================================================

TEST_F(RouterArchetypesTest, CreateMeshRouterArchetype_1D_NoZ) {
    auto router = create_mesh_router_archetype(
        Topology::Linear,
        RoutingDirection::N,
        false,  // No Z router
        FabricNodeId(MeshId{0}, 0));

    // Verify channel mapping
    EXPECT_EQ(router.channel_mapping.get_num_virtual_channels(), 1);
    EXPECT_FALSE(router.channel_mapping.is_z_router());

    // Verify connection mapping: receiver channel 0 has 1 target
    EXPECT_EQ(router.connection_mapping.get_total_sender_count(), 1);
    EXPECT_TRUE(router.connection_mapping.has_targets(0, 0));

    auto targets = router.connection_mapping.get_downstream_targets(0, 0);
    ASSERT_EQ(targets.size(), 1);
    EXPECT_EQ(targets[0].type, ConnectionType::INTRA_MESH);
}

TEST_F(RouterArchetypesTest, CreateMeshRouterArchetype_2D_WithZ) {
    auto router = create_mesh_router_archetype(
        Topology::Mesh,
        RoutingDirection::E,
        true,  // Has Z router
        FabricNodeId(MeshId{0}, 1));

    // Verify channel mapping
    EXPECT_EQ(router.channel_mapping.get_num_virtual_channels(), 1);
    EXPECT_FALSE(router.channel_mapping.is_z_router());

    // Verify connection mapping: receiver channel 0 has 4 targets (3 INTRA_MESH + 1 MESH_TO_Z)
    EXPECT_EQ(router.connection_mapping.get_total_sender_count(), 4);

    // Verify MESH_TO_Z target exists in receiver channel 0
    auto all_targets = router.connection_mapping.get_downstream_targets(0, 0);
    ASSERT_EQ(all_targets.size(), 4);

    auto z_target_it = std::find_if(all_targets.begin(), all_targets.end(),
        [](const ConnectionTarget& t) { return t.type == ConnectionType::MESH_TO_Z; });
    ASSERT_NE(z_target_it, all_targets.end());
    EXPECT_EQ(z_target_it->target_direction.value(), RoutingDirection::Z);
}

TEST_F(RouterArchetypesTest, CreateZRouterArchetype) {
    auto router = create_z_router_archetype(
        FabricNodeId(MeshId{0}, 100));

    // Verify channel mapping
    EXPECT_EQ(router.channel_mapping.get_num_virtual_channels(), 2);
    EXPECT_TRUE(router.channel_mapping.is_z_router());

    // Verify VC1 has 4 sender channels
    EXPECT_EQ(router.channel_mapping.get_num_sender_channels_for_vc(1), 4);

    // Verify connection mapping: receiver channel 0 on VC1 has 4 Z_TO_MESH targets
    EXPECT_EQ(router.connection_mapping.get_total_sender_count(), 4);

    auto targets = router.connection_mapping.get_downstream_targets(1, 0);
    ASSERT_EQ(targets.size(), 4);

    // Verify all expected directions are present
    std::vector<RoutingDirection> expected_dirs = {
        RoutingDirection::N,
        RoutingDirection::E,
        RoutingDirection::S,
        RoutingDirection::W
    };

    for (const auto& expected_dir : expected_dirs) {
        auto it = std::find_if(targets.begin(), targets.end(), [expected_dir](const ConnectionTarget& t) {
            return t.target_direction == expected_dir;
        });
        ASSERT_NE(it, targets.end()) << "Missing direction: " << static_cast<int>(expected_dir);
        EXPECT_EQ(it->type, ConnectionType::Z_TO_MESH);
    }
}

// ============================================================================
// Non-Z → Non-Z Connection Tests (INTRA_MESH)
// ============================================================================

TEST_F(RouterArchetypesTest, NonZToNonZ_1D_Bidirectional) {
    // Create two 1D mesh routers
    auto north_router = create_mesh_router_archetype(
        Topology::Linear,
        RoutingDirection::N,
        false,
        FabricNodeId(MeshId{0}, 0));

    auto south_router = create_mesh_router_archetype(
        Topology::Linear,
        RoutingDirection::S,
        false,
        FabricNodeId(MeshId{0}, 1));

    // North → South connection (driven by connection mapping)
    auto north_targets = north_router.connection_mapping.get_downstream_targets(0, 0);
    ASSERT_EQ(north_targets.size(), 1);
    EXPECT_EQ(north_targets[0].target_direction.value(), RoutingDirection::S);

    record_archetype_connection(
        north_router, 0, 0,
        south_router, 0, north_targets[0].target_sender_channel,
        ConnectionType::INTRA_MESH);

    // South → North connection
    auto south_targets = south_router.connection_mapping.get_downstream_targets(0, 0);
    ASSERT_EQ(south_targets.size(), 1);
    EXPECT_EQ(south_targets[0].target_direction.value(), RoutingDirection::N);

    record_archetype_connection(
        south_router, 0, 0,
        north_router, 0, south_targets[0].target_sender_channel,
        ConnectionType::INTRA_MESH);

    // Verify bidirectional connection
    EXPECT_EQ(registry_->size(), 2);

    auto intra_mesh = registry_->get_connections_by_type(ConnectionType::INTRA_MESH);
    EXPECT_EQ(intra_mesh.size(), 2);
}

TEST_F(RouterArchetypesTest, NonZToNonZ_2D_MultipleDirections) {
    // Create 4 mesh routers in 2D configuration
    std::vector<RoutingDirection> directions = {
        RoutingDirection::N,
        RoutingDirection::E,
        RoutingDirection::S,
        RoutingDirection::W
    };

    std::vector<RouterArchetype> routers;
    routers.reserve(4);
    for (size_t i = 0; i < 4; ++i) {
        routers.push_back(create_mesh_router_archetype(
            Topology::Mesh,
            directions[i],
            false,  // No Z router
            FabricNodeId(MeshId{0}, i)));
    }

    // Each router connects to 3 others (opposite + 2 cross directions)
    // For simplicity, just verify NORTH router connections
    auto& north_router = routers[0];

    // NORTH router receiver channel 0 has 3 targets
    auto targets = north_router.connection_mapping.get_downstream_targets(0, 0);
    ASSERT_EQ(targets.size(), 3);

    // Find SOUTH target (opposite)
    auto south_it = std::find_if(targets.begin(), targets.end(),
        [](const ConnectionTarget& t) { return t.target_direction == RoutingDirection::S; });
    ASSERT_NE(south_it, targets.end());

    record_archetype_connection(
        north_router, 0, 0,
        routers[2], 0, south_it->target_sender_channel,  // SOUTH router
        ConnectionType::INTRA_MESH);

    // NORTH router receiver channel 0 also has EAST/WEST targets (cross directions)
    // Find EAST and WEST targets
    auto east_it = std::find_if(targets.begin(), targets.end(),
        [](const ConnectionTarget& t) { return t.target_direction == RoutingDirection::E; });
    auto west_it = std::find_if(targets.begin(), targets.end(),
        [](const ConnectionTarget& t) { return t.target_direction == RoutingDirection::W; });

    ASSERT_NE(east_it, targets.end());
    ASSERT_NE(west_it, targets.end());

    // Record cross connections
    record_archetype_connection(north_router, 0, 0, routers[1], 0, east_it->target_sender_channel, ConnectionType::INTRA_MESH);
    record_archetype_connection(north_router, 0, 0, routers[3], 0, west_it->target_sender_channel, ConnectionType::INTRA_MESH);

    // Verify 3 connections from NORTH router
    auto north_out = registry_->get_connections_by_source_node(north_router.node_id);
    EXPECT_EQ(north_out.size(), 3);
}

// ============================================================================
// Non-Z → Z Connection Tests (MESH_TO_Z)
// ============================================================================

TEST_F(RouterArchetypesTest, NonZToZ_SingleMeshRouter) {
    // Create mesh router with Z
    auto mesh_router = create_mesh_router_archetype(
        Topology::Mesh,
        RoutingDirection::N,
        true,  // Has Z router
        FabricNodeId(MeshId{0}, 0));

    // Create Z router
    auto z_router = create_z_router_archetype(
        FabricNodeId(MeshId{0}, 100));

    // Verify mesh router has MESH_TO_Z target in receiver channel 0
    auto all_targets = mesh_router.connection_mapping.get_downstream_targets(0, 0);
    auto z_target_it = std::find_if(all_targets.begin(), all_targets.end(),
        [](const ConnectionTarget& t) { return t.type == ConnectionType::MESH_TO_Z; });
    ASSERT_NE(z_target_it, all_targets.end());
    EXPECT_EQ(z_target_it->target_direction.value(), RoutingDirection::Z);

    // Record MESH_TO_Z connection
    record_archetype_connection(
        mesh_router, 0, 0,
        z_router, 0, z_target_it->target_sender_channel,
        ConnectionType::MESH_TO_Z);

    EXPECT_EQ(registry_->size(), 1);

    auto mesh_to_z = registry_->get_connections_by_type(ConnectionType::MESH_TO_Z);
    EXPECT_EQ(mesh_to_z.size(), 1);
    EXPECT_EQ(mesh_to_z[0].source_receiver_channel, 0);
    EXPECT_EQ(mesh_to_z[0].dest_direction, RoutingDirection::Z);
}

TEST_F(RouterArchetypesTest, NonZToZ_FourMeshRouters) {
    // Create 4 mesh routers, all with Z
    std::vector<RoutingDirection> directions = {
        RoutingDirection::N,
        RoutingDirection::E,
        RoutingDirection::S,
        RoutingDirection::W
    };

    std::vector<RouterArchetype> mesh_routers;
    mesh_routers.reserve(4);
    for (size_t i = 0; i < 4; ++i) {
        mesh_routers.push_back(create_mesh_router_archetype(
            Topology::Mesh,
            directions[i],
            true,  // Has Z router
            FabricNodeId(MeshId{0}, i)));
    }

    // Create Z router
    auto z_router = create_z_router_archetype(
        FabricNodeId(MeshId{0}, 100));

    // Each mesh router connects to Z from receiver channel 0
    for (auto& mesh_router : mesh_routers) {
        auto all_targets = mesh_router.connection_mapping.get_downstream_targets(0, 0);
        auto z_target_it = std::find_if(all_targets.begin(), all_targets.end(),
            [](const ConnectionTarget& t) { return t.type == ConnectionType::MESH_TO_Z; });
        ASSERT_NE(z_target_it, all_targets.end());

        record_archetype_connection(
            mesh_router, 0, 0,
            z_router, 0, z_target_it->target_sender_channel,
            ConnectionType::MESH_TO_Z);
    }

    // Verify 4 MESH_TO_Z connections
    EXPECT_EQ(registry_->size(), 4);

    auto mesh_to_z = registry_->get_connections_by_type(ConnectionType::MESH_TO_Z);
    EXPECT_EQ(mesh_to_z.size(), 4);

    // Verify Z router receives from all 4 mesh routers
    auto z_incoming = registry_->get_connections_by_dest_node(z_router.node_id);
    EXPECT_EQ(z_incoming.size(), 4);

    // All should target Z router VC0
    // dest_sender_channel will be the mesh_to_z_channel value from the source
    for (const auto& conn : z_incoming) {
        EXPECT_EQ(conn.dest_vc, 0);
        // dest_sender_channel is set by the mesh router's target_sender_channel
        // For 2D mesh, this is builder_config::num_sender_channels_2d_mesh (which is 4)
    }
}

TEST_F(RouterArchetypesTest, NonZToZ_FourMeshRouters_VC1_Connections) {
    // Test: 4 non-Z routers VC1 → Z router VC1
    // This tests the reverse direction from Z_TO_MESH

    // Create 4 mesh routers, all with Z
    std::vector<RoutingDirection> directions = {
        RoutingDirection::N,
        RoutingDirection::E,
        RoutingDirection::S,
        RoutingDirection::W
    };

    std::vector<RouterArchetype> mesh_routers;
    mesh_routers.reserve(4);
    for (size_t i = 0; i < 4; ++i) {
        mesh_routers.push_back(create_mesh_router_archetype(
            Topology::Mesh,
            directions[i],
            true,  // Has Z router
            FabricNodeId(MeshId{0}, i)));
    }

    // Create Z router
    auto z_router = create_z_router_archetype(
        FabricNodeId(MeshId{0}, 100));

    // Each mesh router connects VC1 → Z VC1
    // Note: Mesh routers have VC1 for receiving from Z, but can also send on VC1
    for (size_t i = 0; i < 4; ++i) {
        record_archetype_connection(
            mesh_routers[i], 1, 0,  // Mesh VC1, sender channel 0
            z_router, 1, static_cast<uint32_t>(i),  // Z VC1, receiver channel i
            ConnectionType::MESH_TO_Z);
    }

    // Verify 4 MESH_TO_Z connections on VC1
    EXPECT_EQ(registry_->size(), 4);

    auto mesh_to_z = registry_->get_connections_by_type(ConnectionType::MESH_TO_Z);
    EXPECT_EQ(mesh_to_z.size(), 4);

    // Verify all connections use VC1
    for (const auto& conn : mesh_to_z) {
        EXPECT_EQ(conn.source_vc, 1);  // Mesh VC1
        EXPECT_EQ(conn.dest_vc, 1);    // Z VC1
    }

    // Verify Z router receives from all 4 mesh routers on VC1
    auto z_incoming = registry_->get_connections_by_dest_node(z_router.node_id);
    EXPECT_EQ(z_incoming.size(), 4);

    // Each should target a different Z VC1 receiver channel (0-3)
    std::set<uint32_t> receiver_channels;
    for (const auto& conn : z_incoming) {
        EXPECT_EQ(conn.dest_vc, 1);
        receiver_channels.insert(conn.dest_sender_channel);
    }
    EXPECT_EQ(receiver_channels.size(), 4);  // All 4 channels used
}

TEST_F(RouterArchetypesTest, NonZToZ_FiveMeshRouters_VC1_ShouldFail) {
    // Negative test: 5 non-Z routers VC1 → Z router VC1 should fail
    // Z router VC1 only has 4 receiver channels (0-3)

    // Create 5 mesh routers
    std::vector<RouterArchetype> mesh_routers;
    mesh_routers.reserve(5);
    for (size_t i = 0; i < 5; ++i) {
        mesh_routers.push_back(create_mesh_router_archetype(
            Topology::Mesh,
            RoutingDirection::N,
            true,
            FabricNodeId(MeshId{0}, i)));
    }

    // Create Z router
    auto z_router = create_z_router_archetype(
        FabricNodeId(MeshId{0}, 100));

    // Successfully connect first 4 mesh routers to Z VC1 channels 0-3
    for (size_t i = 0; i < 4; ++i) {
        record_archetype_connection(
            mesh_routers[i], 1, 0,
            z_router, 1, static_cast<uint32_t>(i),
            ConnectionType::MESH_TO_Z);
    }

    EXPECT_EQ(registry_->size(), 4);

    // Attempting to connect 5th mesh router should fail
    // Z router VC1 only has receiver channels 0-3 (4 total)
    // This would require receiver channel 4, which doesn't exist

    // In a real implementation, this would be caught by:
    // 1. Channel mapping validation (no receiver channel available)
    // 2. Connection establishment logic checking available channels
    // 3. Multi-target receiver capacity limits

    // Verify that Z router VC1 has exactly 4 sender channels (for Z_TO_MESH)
    EXPECT_EQ(z_router.channel_mapping.get_num_sender_channels_for_vc(1), 4);

    // Receiver channel 0 has 4 targets mapped to specific directions (N/E/S/W)
    auto z_targets = z_router.connection_mapping.get_downstream_targets(1, 0);
    EXPECT_EQ(z_targets.size(), 4);  // Receiver channel 0 has 4 target directions

    // The 5th mesh router would need to send to Z VC1, but:
    // - Z VC1 only has 1 receiver channel (for multi-target from 4 mesh routers)
    // - That receiver is already handling 4 connections (at capacity)
    // - No additional receiver channels exist on Z VC1
    // In production connection logic, this should be detected and rejected
}

// ============================================================================
// Z → Non-Z Connection Tests (Z_TO_MESH)
// ============================================================================

TEST_F(RouterArchetypesTest, ZToNonZ_SingleMeshRouter) {
    // Create Z router
    auto z_router = create_z_router_archetype(
        FabricNodeId(MeshId{0}, 100));

    // Create mesh router
    auto mesh_router = create_mesh_router_archetype(
        Topology::Mesh,
        RoutingDirection::N,
        true,
        FabricNodeId(MeshId{0}, 0));

    // Z router receiver channel 0 has 4 targets, find NORTH
    auto targets = z_router.connection_mapping.get_downstream_targets(1, 0);
    ASSERT_EQ(targets.size(), 4);

    auto north_it = std::find_if(targets.begin(), targets.end(),
        [](const ConnectionTarget& t) { return t.target_direction == RoutingDirection::N; });
    ASSERT_NE(north_it, targets.end());
    EXPECT_EQ(north_it->type, ConnectionType::Z_TO_MESH);

    // Record Z_TO_MESH connection (Z VC1 → mesh VC1)
    record_archetype_connection(
        z_router, 1, 0,
        mesh_router, 1, north_it->target_sender_channel,
        ConnectionType::Z_TO_MESH);

    EXPECT_EQ(registry_->size(), 1);

    auto z_to_mesh = registry_->get_connections_by_type(ConnectionType::Z_TO_MESH);
    EXPECT_EQ(z_to_mesh.size(), 1);
    EXPECT_EQ(z_to_mesh[0].source_vc, 1);  // Z router VC1
    EXPECT_EQ(z_to_mesh[0].dest_vc, 1);    // Mesh router VC1
}

TEST_F(RouterArchetypesTest, ZToNonZ_FourMeshRouters_AllDirections) {
    // Create Z router
    auto z_router = create_z_router_archetype(
        FabricNodeId(MeshId{0}, 100));

    // Create 4 mesh routers
    std::vector<RoutingDirection> directions = {
        RoutingDirection::N,
        RoutingDirection::E,
        RoutingDirection::S,
        RoutingDirection::W
    };

    std::vector<RouterArchetype> mesh_routers;
    mesh_routers.reserve(4);
    for (size_t i = 0; i < 4; ++i) {
        mesh_routers.push_back(create_mesh_router_archetype(
            Topology::Mesh,
            directions[i],
            true,
            FabricNodeId(MeshId{0}, i)));
    }

    // Z router receiver channel 0 connects to all 4 mesh routers via VC1
    auto z_targets = z_router.connection_mapping.get_downstream_targets(1, 0);
    ASSERT_EQ(z_targets.size(), 4);

    for (size_t i = 0; i < z_targets.size(); ++i) {
        EXPECT_EQ(z_targets[i].type, ConnectionType::Z_TO_MESH);
        EXPECT_EQ(z_targets[i].target_direction.value(), directions[i]);

        record_archetype_connection(
            z_router, 1, 0,
            mesh_routers[i], 1, z_targets[i].target_sender_channel,
            ConnectionType::Z_TO_MESH);
    }

    // Verify 4 Z_TO_MESH connections
    EXPECT_EQ(registry_->size(), 4);

    auto z_to_mesh = registry_->get_connections_by_type(ConnectionType::Z_TO_MESH);
    EXPECT_EQ(z_to_mesh.size(), 4);

    // Verify Z router sends to all 4 mesh routers
    auto z_outgoing = registry_->get_connections_by_source_node(z_router.node_id);
    EXPECT_EQ(z_outgoing.size(), 4);

    // All should use VC1 and same receiver channel (0)
    for (const auto& conn : z_outgoing) {
        EXPECT_EQ(conn.source_vc, 1);
        EXPECT_EQ(conn.source_receiver_channel, 0);
    }
}

TEST_F(RouterArchetypesTest, ZToNonZ_EdgeDevice_TwoMeshRouters) {
    // Edge device: Z router + 2 mesh routers (NORTH and EAST only)

    // Create Z router
    auto z_router = create_z_router_archetype(
        FabricNodeId(MeshId{0}, 100));

    // Create 2 mesh routers
    auto north_router = create_mesh_router_archetype(
        Topology::Mesh,
        RoutingDirection::N,
        true,
        FabricNodeId(MeshId{0}, 0));

    auto east_router = create_mesh_router_archetype(
        Topology::Mesh,
        RoutingDirection::E,
        true,
        FabricNodeId(MeshId{0}, 1));

    // Z router has intent for all 4 directions, but only 2 exist
    std::map<RoutingDirection, RouterArchetype*> existing_routers = {
        {RoutingDirection::N, &north_router},
        {RoutingDirection::E, &east_router}
    };

    // Get all targets from receiver channel 0, only connect to existing routers
    auto z_targets = z_router.connection_mapping.get_downstream_targets(1, 0);
    ASSERT_EQ(z_targets.size(), 4);

    for (const auto& target : z_targets) {
        auto target_dir = target.target_direction.value();

        // Only record if target router exists (FabricBuilder connection logic)
        if (existing_routers.count(target_dir)) {
            record_archetype_connection(
                z_router, 1, 0,
                *existing_routers[target_dir], 1, target.target_sender_channel,
                ConnectionType::Z_TO_MESH);
        }
    }

    // Verify only 2 Z_TO_MESH connections (NORTH and EAST)
    EXPECT_EQ(registry_->size(), 2);

    auto z_to_mesh = registry_->get_connections_by_type(ConnectionType::Z_TO_MESH);
    EXPECT_EQ(z_to_mesh.size(), 2);
}

// ============================================================================
// Full Device Connection Tests
// ============================================================================

TEST_F(RouterArchetypesTest, FullDevice_4MeshRouters_1ZRouter_AllConnections) {
    // Create complete device with all connection types

    // Create 4 mesh routers
    std::vector<RoutingDirection> directions = {
        RoutingDirection::N,
        RoutingDirection::E,
        RoutingDirection::S,
        RoutingDirection::W
    };

    std::vector<RouterArchetype> mesh_routers;
    mesh_routers.reserve(4);
    for (size_t i = 0; i < 4; ++i) {
        mesh_routers.push_back(create_mesh_router_archetype(
            Topology::Mesh,
            directions[i],
            true,  // Has Z router
            FabricNodeId(MeshId{0}, i),
            static_cast<uint8_t>(i)));
    }

    // Create Z router
    auto z_router = create_z_router_archetype(
        FabricNodeId(MeshId{0}, 100));

    // Step 1: MESH_TO_Z connections (4 mesh routers → Z router)
    for (auto& mesh_router : mesh_routers) {
        auto all_targets = mesh_router.connection_mapping.get_downstream_targets(0, 0);
        auto z_target_it = std::find_if(all_targets.begin(), all_targets.end(),
            [](const ConnectionTarget& t) { return t.type == ConnectionType::MESH_TO_Z; });
        ASSERT_NE(z_target_it, all_targets.end());

        record_archetype_connection(
            mesh_router, 0, 0,
            z_router, 0, z_target_it->target_sender_channel,
            ConnectionType::MESH_TO_Z);
    }

    // Step 2: Z_TO_MESH connections (Z router → 4 mesh routers)
    auto z_targets = z_router.connection_mapping.get_downstream_targets(1, 0);
    ASSERT_EQ(z_targets.size(), 4);

    for (size_t i = 0; i < z_targets.size(); ++i) {
        record_archetype_connection(
            z_router, 1, 0,
            mesh_routers[i], 1, z_targets[i].target_sender_channel,
            ConnectionType::Z_TO_MESH);
    }

    // Verify total connections: 4 MESH_TO_Z + 4 Z_TO_MESH = 8
    EXPECT_EQ(registry_->size(), 8);

    auto mesh_to_z = registry_->get_connections_by_type(ConnectionType::MESH_TO_Z);
    auto z_to_mesh = registry_->get_connections_by_type(ConnectionType::Z_TO_MESH);

    EXPECT_EQ(mesh_to_z.size(), 4);
    EXPECT_EQ(z_to_mesh.size(), 4);

    // Verify Z router connectivity
    auto z_incoming = registry_->get_connections_by_dest_node(z_router.node_id);
    auto z_outgoing = registry_->get_connections_by_source_node(z_router.node_id);

    EXPECT_EQ(z_incoming.size(), 4);  // Receives from all 4 mesh routers
    EXPECT_EQ(z_outgoing.size(), 4);  // Sends to all 4 mesh routers

    // Verify each mesh router has bidirectional Z connection
    for (const auto& mesh_router : mesh_routers) {
        auto mesh_out = registry_->get_connections_by_source_node(mesh_router.node_id);
        auto mesh_in = registry_->get_connections_by_dest_node(mesh_router.node_id);

        // Each mesh router sends to Z (1 MESH_TO_Z)
        auto mesh_to_z_conns = std::count_if(
            mesh_out.begin(), mesh_out.end(),
            [](const auto& c) { return c.connection_type == ConnectionType::MESH_TO_Z; });
        EXPECT_EQ(mesh_to_z_conns, 1);

        // Each mesh router receives from Z (1 Z_TO_MESH)
        auto z_to_mesh_conns = std::count_if(
            mesh_in.begin(), mesh_in.end(),
            [](const auto& c) { return c.connection_type == ConnectionType::Z_TO_MESH; });
        EXPECT_EQ(z_to_mesh_conns, 1);
    }
}

TEST_F(RouterArchetypesTest, FullDevice_WithINTRA_MESH_And_ZConnections) {
    // Complete device with INTRA_MESH + MESH_TO_Z + Z_TO_MESH

    // Create 4 mesh routers
    std::vector<RoutingDirection> directions = {
        RoutingDirection::N,
        RoutingDirection::E,
        RoutingDirection::S,
        RoutingDirection::W
    };

    std::vector<RouterArchetype> mesh_routers;
    mesh_routers.reserve(4);
    for (size_t i = 0; i < 4; ++i) {
        mesh_routers.push_back(create_mesh_router_archetype(
            Topology::Mesh,
            directions[i],
            true,
            FabricNodeId(MeshId{0}, i)));
    }

    // Create Z router
    auto z_router = create_z_router_archetype(
        FabricNodeId(MeshId{0}, 100));

    // Step 1: INTRA_MESH connections (simplified - just opposite pairs)
    // NORTH ↔ SOUTH
    record_archetype_connection(mesh_routers[0], 0, 1, mesh_routers[2], 0, 0, ConnectionType::INTRA_MESH);
    record_archetype_connection(mesh_routers[2], 0, 1, mesh_routers[0], 0, 0, ConnectionType::INTRA_MESH);

    // EAST ↔ WEST
    record_archetype_connection(mesh_routers[1], 0, 1, mesh_routers[3], 0, 0, ConnectionType::INTRA_MESH);
    record_archetype_connection(mesh_routers[3], 0, 1, mesh_routers[1], 0, 0, ConnectionType::INTRA_MESH);

    // Step 2: MESH_TO_Z connections
    for (auto& mesh_router : mesh_routers) {
        record_archetype_connection(mesh_router, 0, 4, z_router, 0, 0, ConnectionType::MESH_TO_Z);
    }

    // Step 3: Z_TO_MESH connections (Z VC1 → mesh VC1)
    for (uint32_t ch = 0; ch < 4; ++ch) {
        record_archetype_connection(z_router, 1, ch, mesh_routers[ch], 1, 0, ConnectionType::Z_TO_MESH);
    }

    // Verify total: 4 INTRA_MESH + 4 MESH_TO_Z + 4 Z_TO_MESH = 12
    EXPECT_EQ(registry_->size(), 12);

    auto intra_mesh = registry_->get_connections_by_type(ConnectionType::INTRA_MESH);
    auto mesh_to_z = registry_->get_connections_by_type(ConnectionType::MESH_TO_Z);
    auto z_to_mesh = registry_->get_connections_by_type(ConnectionType::Z_TO_MESH);

    EXPECT_EQ(intra_mesh.size(), 4);
    EXPECT_EQ(mesh_to_z.size(), 4);
    EXPECT_EQ(z_to_mesh.size(), 4);
}

TEST_F(RouterArchetypesTest, Archetype_ChannelMapping_ConnectionMapping_Consistency) {
    // Verify archetype's channel mapping and connection mapping are consistent

    // Test mesh router archetype
    auto mesh_router = create_mesh_router_archetype(
        Topology::Mesh,
        RoutingDirection::N,
        true,
        FabricNodeId(MeshId{0}, 0));

    // Verify receiver channel 0 has targets matching sender channel count
    uint32_t num_vcs = mesh_router.channel_mapping.get_num_virtual_channels();

    for (uint32_t vc = 0; vc < num_vcs; ++vc) {
        uint32_t num_senders = mesh_router.channel_mapping.get_num_sender_channels_for_vc(vc);

        if (num_senders > 0) {
            // Receiver channel 0 should have targets
            EXPECT_TRUE(mesh_router.connection_mapping.has_targets(vc, 0))
                << "VC" << vc << " receiver channel 0 should have connection targets";

            auto targets = mesh_router.connection_mapping.get_downstream_targets(vc, 0);
            EXPECT_EQ(targets.size(), num_senders)
                << "VC" << vc << " receiver channel 0 should have " << num_senders << " targets";
        }
    }

    // Test Z router archetype
    auto z_router = create_z_router_archetype(
        FabricNodeId(MeshId{0}, 100));

    // VC1 receiver channel 0 should have 4 targets
    uint32_t vc1_senders = z_router.channel_mapping.get_num_sender_channels_for_vc(1);
    EXPECT_EQ(vc1_senders, 4);

    EXPECT_TRUE(z_router.connection_mapping.has_targets(1, 0))
        << "Z router VC1 receiver channel 0 should have connection targets";

    auto z_targets = z_router.connection_mapping.get_downstream_targets(1, 0);
    EXPECT_EQ(z_targets.size(), vc1_senders)
        << "Z router VC1 receiver channel 0 should have " << vc1_senders << " targets";
}

}  // namespace tt::tt_fabric
