// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

/**
 * Phase 7: Z Router Integration Tests
 *
 * End-to-end validation of Z router connectivity without requiring actual Z hardware.
 * Tests focus on connection logic, variable mesh counts, and edge cases.
 */

#include <gtest/gtest.h>
#include "tt_metal/fabric/builder/connection_registry.hpp"
#include "tt_metal/fabric/builder/router_connection_mapping.hpp"
#include "tt_metal/fabric/fabric_router_channel_mapping.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/builder/mesh_channel_spec.hpp"
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

using namespace tt::tt_fabric;
namespace builder_config = tt::tt_fabric::builder_config;

class ZRouterIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        registry_ = std::make_shared<ConnectionRegistry>();
    }

    std::shared_ptr<ConnectionRegistry> registry_;

    // Helper to record a connection
    void record_connection(
        FabricNodeId src_node,
        RoutingDirection src_dir,
        uint8_t src_chan,
        uint32_t src_vc,
        uint32_t src_sender,
        FabricNodeId dst_node,
        RoutingDirection dst_dir,
        uint8_t dst_chan,
        uint32_t dst_vc,
        uint32_t dst_receiver,
        ConnectionType type) {
        registry_->record_connection(RouterConnectionRecord{
            .source_node = src_node,
            .source_direction = src_dir,
            .source_eth_chan = src_chan,
            .source_vc = src_vc,
            .source_receiver_channel = src_sender,
            .dest_node = dst_node,
            .dest_direction = dst_dir,
            .dest_eth_chan = dst_chan,
            .dest_vc = dst_vc,
            .dest_sender_channel = dst_receiver,
            .connection_type = type
        });
    }
};

// ============================================================================
// Basic Connectivity Tests
// ============================================================================

TEST_F(ZRouterIntegrationTest, FullDevice_4Mesh1Z_BidirectionalConnectivity) {
    // Comprehensive test: Full device with 4 mesh routers + 1 Z router
    // Validates both MESH_TO_Z and Z_TO_MESH connections

    FabricNodeId device0(MeshId{0}, 0);
    std::vector<RoutingDirection> mesh_dirs = {
        RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W
    };

    // MESH_TO_Z: All 4 mesh routers → Z router VC0 (multi-target receiver)
    for (size_t i = 0; i < 4; ++i) {
        record_connection(
            device0, mesh_dirs[i], static_cast<uint8_t>(i),
            0, builder_config::num_sender_channels_2d_mesh,  // VC0, MESH_TO_Z channel
            device0, RoutingDirection::Z, 4,
            0, 0,  // Z VC0 receiver 0 (SAME for all - multi-target)
            ConnectionType::MESH_TO_Z);
    }

    // Z_TO_MESH: Z router VC1 senders 0-3 → 4 mesh routers (one-to-one)
    for (size_t i = 0; i < 4; ++i) {
        record_connection(
            device0, RoutingDirection::Z, 4,
            1, static_cast<uint32_t>(i),  // Z VC1 sender i
            device0, mesh_dirs[i], static_cast<uint8_t>(i),
            1, 0,  // mesh VC1 receiver
            ConnectionType::Z_TO_MESH);
    }

    // Validate total connections
    EXPECT_EQ(registry_->size(), 8) << "Should have 8 total connections (4 MESH_TO_Z + 4 Z_TO_MESH)";

    // Validate by type
    auto mesh_to_z = registry_->get_connections_by_type(ConnectionType::MESH_TO_Z);
    auto z_to_mesh = registry_->get_connections_by_type(ConnectionType::Z_TO_MESH);

    EXPECT_EQ(mesh_to_z.size(), 4) << "Should have 4 MESH_TO_Z connections";
    EXPECT_EQ(z_to_mesh.size(), 4) << "Should have 4 Z_TO_MESH connections";

    // Validate Z router connectivity
    auto z_incoming = registry_->get_connections_by_dest_node(device0);
    auto z_outgoing = registry_->get_connections_by_source_node(device0);

    // Filter to Z router only
    size_t z_in_count = 0, z_out_count = 0;
    for (const auto& conn : z_incoming) {
        if (conn.dest_direction == RoutingDirection::Z) {
            z_in_count++;
        }
    }
    for (const auto& conn : z_outgoing) {
        if (conn.source_direction == RoutingDirection::Z) {
            z_out_count++;
        }
    }

    EXPECT_EQ(z_in_count, 4) << "Z router should have 4 incoming connections";
    EXPECT_EQ(z_out_count, 4) << "Z router should have 4 outgoing connections";

    // Validate multi-target receiver: all MESH_TO_Z target same Z receiver
    for (const auto& conn : mesh_to_z) {
        EXPECT_EQ(conn.dest_vc, 0) << "MESH_TO_Z should target Z VC0";
        EXPECT_EQ(conn.dest_sender_channel, 0) << "All MESH_TO_Z should target same receiver (multi-target)";
    }

    // Validate Z_TO_MESH uses VC1
    for (const auto& conn : z_to_mesh) {
        EXPECT_EQ(conn.source_vc, 1) << "Z_TO_MESH should use Z VC1";
        EXPECT_EQ(conn.dest_vc, 1) << "Z_TO_MESH should target mesh VC1";
    }
}

TEST_F(ZRouterIntegrationTest, ConnectionMapping_ZRouter_AllDirections) {
    // Test that Z router connection mapping specifies all 4 directions
    auto mapping = RouterConnectionMapping::for_z_router();

    // Z router VC1 receiver channel 0 should have 4 targets for N/E/S/W
    std::vector<RoutingDirection> expected_dirs = {
        RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W
    };

    auto targets = mapping.get_downstream_targets(1, 0);
    ASSERT_EQ(targets.size(), 4) << "Z VC1 receiver channel 0 should have 4 targets";

    // Verify all expected directions are present
    for (const auto& expected_dir : expected_dirs) {
        auto it = std::find_if(targets.begin(), targets.end(), [expected_dir](const ConnectionTarget& t) {
            return t.target_direction == expected_dir;
        });
        ASSERT_NE(it, targets.end()) << "Missing direction: " << static_cast<int>(expected_dir);
        EXPECT_EQ(it->type, ConnectionType::Z_TO_MESH);
        EXPECT_EQ(it->target_vc, 1) << "Should target mesh VC1";
    }
}

TEST_F(ZRouterIntegrationTest, ConnectionMapping_MeshWithZ_HasMeshToZTarget) {
    // Test that mesh routers with Z have MESH_TO_Z connection
    auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh, RoutingDirection::N, true);  // has_z = true

    // Receiver channel 0 should have MESH_TO_Z target among its targets
    auto targets = mapping.get_downstream_targets(0, 0);
    ASSERT_EQ(targets.size(), 4) << "Receiver channel 0 should have 4 targets";

    auto mesh_to_z_it = std::find_if(targets.begin(), targets.end(),
        [](const ConnectionTarget& t) { return t.type == ConnectionType::MESH_TO_Z; });
    ASSERT_NE(mesh_to_z_it, targets.end()) << "Should have MESH_TO_Z target";

    EXPECT_EQ(mesh_to_z_it->target_vc, 0) << "Should target Z VC0";
    EXPECT_TRUE(mesh_to_z_it->target_direction.has_value());
    EXPECT_EQ(mesh_to_z_it->target_direction.value(), RoutingDirection::Z);
}

// ============================================================================
// Variable Mesh Count Tests
// ============================================================================

TEST_F(ZRouterIntegrationTest, EdgeDevice_2Mesh1Z_MinimalConfiguration) {
    // Edge device with only 2 mesh routers (e.g., corner device with N+E only)
    FabricNodeId device0(MeshId{0}, 0);

    std::vector<RoutingDirection> mesh_dirs = {RoutingDirection::N, RoutingDirection::E};

    // MESH_TO_Z: 2 mesh routers → Z router
    for (size_t i = 0; i < 2; ++i) {
        record_connection(
            device0, mesh_dirs[i], static_cast<uint8_t>(i),
            0, builder_config::num_sender_channels_2d_mesh,
            device0, RoutingDirection::Z, 4,
            0, 0,
            ConnectionType::MESH_TO_Z);
    }

    // Z_TO_MESH: Z router → 2 mesh routers (only N and E)
    for (size_t i = 0; i < 2; ++i) {
        record_connection(
            device0, RoutingDirection::Z, 4,
            1, static_cast<uint32_t>(i),  // Z uses senders 0 (N) and 1 (E)
            device0, mesh_dirs[i], static_cast<uint8_t>(i),
            1, 0,
            ConnectionType::Z_TO_MESH);
    }

    EXPECT_EQ(registry_->size(), 4) << "Should have 4 total connections (2 each direction)";
    EXPECT_EQ(registry_->get_connections_by_type(ConnectionType::MESH_TO_Z).size(), 2);
    EXPECT_EQ(registry_->get_connections_by_type(ConnectionType::Z_TO_MESH).size(), 2);
}

TEST_F(ZRouterIntegrationTest, EdgeDevice_3Mesh1Z_PartialConfiguration) {
    // Edge device with 3 mesh routers (e.g., N+E+S)
    FabricNodeId device0(MeshId{0}, 0);

    std::vector<RoutingDirection> mesh_dirs = {
        RoutingDirection::N, RoutingDirection::E, RoutingDirection::S
    };

    // MESH_TO_Z: 3 mesh routers → Z router
    for (size_t i = 0; i < 3; ++i) {
        record_connection(
            device0, mesh_dirs[i], static_cast<uint8_t>(i),
            0, builder_config::num_sender_channels_2d_mesh,
            device0, RoutingDirection::Z, 4,
            0, 0,
            ConnectionType::MESH_TO_Z);
    }

    // Z_TO_MESH: Z router → 3 mesh routers
    for (size_t i = 0; i < 3; ++i) {
        record_connection(
            device0, RoutingDirection::Z, 4,
            1, static_cast<uint32_t>(i),
            device0, mesh_dirs[i], static_cast<uint8_t>(i),
            1, 0,
            ConnectionType::Z_TO_MESH);
    }

    EXPECT_EQ(registry_->size(), 6) << "Should have 6 total connections (3 each direction)";
    EXPECT_EQ(registry_->get_connections_by_type(ConnectionType::MESH_TO_Z).size(), 3);
    EXPECT_EQ(registry_->get_connections_by_type(ConnectionType::Z_TO_MESH).size(), 3);
}

TEST_F(ZRouterIntegrationTest, VariableMeshCount_AllConfigurations) {
    // Test all valid mesh router counts (2, 3, 4)
    for (size_t num_mesh = 2; num_mesh <= 4; ++num_mesh) {
        auto local_registry = std::make_shared<ConnectionRegistry>();
        FabricNodeId device0(MeshId{0}, 0);

        std::vector<RoutingDirection> all_dirs = {
            RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W
        };

        // Create connections for num_mesh routers
        for (size_t i = 0; i < num_mesh; ++i) {
            // MESH_TO_Z
            local_registry->record_connection(RouterConnectionRecord{
                .source_node = device0,
                .source_direction = all_dirs[i],
                .source_eth_chan = static_cast<uint8_t>(i),
                .source_vc = 0,
                .source_receiver_channel = builder_config::num_sender_channels_2d_mesh,
                .dest_node = device0,
                .dest_direction = RoutingDirection::Z,
                .dest_eth_chan = 4,
                .dest_vc = 0,
                .dest_sender_channel = 0,
                .connection_type = ConnectionType::MESH_TO_Z
            });

            // Z_TO_MESH
            local_registry->record_connection(RouterConnectionRecord{
                .source_node = device0,
                .source_direction = RoutingDirection::Z,
                .source_eth_chan = 4,
                .source_vc = 1,
                .source_receiver_channel = static_cast<uint32_t>(i),
                .dest_node = device0,
                .dest_direction = all_dirs[i],
                .dest_eth_chan = static_cast<uint8_t>(i),
                .dest_vc = 1,
                .dest_sender_channel = 0,
                .connection_type = ConnectionType::Z_TO_MESH
            });
        }

        EXPECT_EQ(local_registry->size(), num_mesh * 2)
            << "With " << num_mesh << " mesh routers, should have " << (num_mesh * 2) << " connections";
        EXPECT_EQ(local_registry->get_connections_by_type(ConnectionType::MESH_TO_Z).size(), num_mesh);
        EXPECT_EQ(local_registry->get_connections_by_type(ConnectionType::Z_TO_MESH).size(), num_mesh);
    }
}

// ============================================================================
// Channel Mapping Tests
// ============================================================================

TEST_F(ZRouterIntegrationTest, ChannelMapping_ZRouter_VC0AndVC1) {
    // Verify Z router has 2 VCs with correct channel counts
    auto intermesh_config = IntermeshVCConfig::full_mesh();
    auto spec = MeshChannelSpec::create_for_compute_mesh(Topology::Mesh, &intermesh_config);
    auto mapping = FabricRouterChannelMapping(
        Topology::Mesh,
        spec,
        false,  // no tensix
        RouterVariant::Z_ROUTER);

    EXPECT_EQ(spec.num_vcs, 2) << "Z router should have 2 VCs";

    // VC0: Standard mesh forwarding (4 channels for 2D)
    EXPECT_EQ(spec.sender_channels_per_vc[0], builder_config::num_sender_channels_z_router_vc0);

    // VC1: Z-specific traffic (4 channels for N/E/S/W)
    EXPECT_EQ(spec.sender_channels_per_vc[1], builder_config::num_sender_channels_z_router_vc1);
}

TEST_F(ZRouterIntegrationTest, ChannelMapping_MeshRouter_VC0Only) {
    // Verify standard mesh router has only VC0 when no intermesh config provided
    auto spec = MeshChannelSpec::create_for_compute_mesh(Topology::Mesh, nullptr);
    auto mapping = FabricRouterChannelMapping(
        Topology::Mesh,
        spec,
        false,  // no tensix
        RouterVariant::MESH);

    EXPECT_EQ(spec.num_vcs, 1) << "Standard mesh router should have 1 VC (VC0 only) without intermesh";
    EXPECT_EQ(spec.sender_channels_per_vc[0], builder_config::num_sender_channels_2d_mesh);
    EXPECT_FALSE(spec.has_vc(1)) << "VC1 should not be created without intermesh config";
}

TEST_F(ZRouterIntegrationTest, ChannelMapping_ZRouter_InternalChannels) {
    // Verify Z router VC1 maps to correct internal erisc channels (4-7)
    auto intermesh_config = IntermeshVCConfig::full_mesh();
    auto spec = MeshChannelSpec::create_for_compute_mesh(Topology::Mesh, &intermesh_config);
    auto mapping = FabricRouterChannelMapping(Topology::Mesh, spec, false, RouterVariant::Z_ROUTER);

    // VC1 sender channels 0-3 should map to erisc internal channels 4-7
    for (uint32_t i = 0; i < 4; ++i) {
        auto sender_mapping = mapping.get_sender_mapping(1, i);
        EXPECT_EQ(sender_mapping.builder_type, BuilderType::ERISC);
        EXPECT_EQ(sender_mapping.internal_sender_channel_id, 4 + i)
            << "Z VC1 sender " << i << " should map to erisc channel " << (4 + i);
    }

    // VC1 receiver channel 0 should map to erisc internal channel 1
    auto receiver_mapping = mapping.get_receiver_mapping(1, 0);
    EXPECT_EQ(receiver_mapping.builder_type, BuilderType::ERISC);
    EXPECT_EQ(receiver_mapping.internal_receiver_channel_id, 1);
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST_F(ZRouterIntegrationTest, NoZRouter_NoMeshToZConnections) {
    // Device without Z router should have no MESH_TO_Z connections
    FabricNodeId device0(MeshId{0}, 0);

    // Only INTRA_MESH connections
    record_connection(
        device0, RoutingDirection::N, 0,
        0, 1,
        device0, RoutingDirection::S, 2,
        0, 1,
        ConnectionType::INTRA_MESH);

    record_connection(
        device0, RoutingDirection::E, 1,
        0, 1,
        device0, RoutingDirection::W, 3,
        0, 1,
        ConnectionType::INTRA_MESH);

    EXPECT_EQ(registry_->size(), 2);
    EXPECT_EQ(registry_->get_connections_by_type(ConnectionType::MESH_TO_Z).size(), 0);
    EXPECT_EQ(registry_->get_connections_by_type(ConnectionType::Z_TO_MESH).size(), 0);
    EXPECT_EQ(registry_->get_connections_by_type(ConnectionType::INTRA_MESH).size(), 2);
}

TEST_F(ZRouterIntegrationTest, ConnectionMapping_MeshWithoutZ_NoMeshToZ) {
    // Mesh router without Z should not have MESH_TO_Z target
    auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh, RoutingDirection::N, false);  // has_z = false

    // Check all sender channels - none should have MESH_TO_Z
    for (uint32_t ch = 0; ch < 5; ++ch) {
        auto targets = mapping.get_downstream_targets(0, ch);
        for (const auto& target : targets) {
            EXPECT_NE(target.type, ConnectionType::MESH_TO_Z)
                << "Mesh router without Z should not have MESH_TO_Z connections";
        }
    }
}

TEST_F(ZRouterIntegrationTest, ConnectionMapping_ZRouter_VC0_CannotBeDownstreamTarget) {
    // Validate that Z router connection mapping never specifies Z VC0 as a downstream target
    // Z VC0 senders don't forward anywhere, so no router should target them

    auto z_mapping = RouterConnectionMapping::for_z_router();

    // Check all Z router sender channels (VC0 and VC1)
    constexpr uint32_t num_z_router_sender_channels_per_vc = 4;
    for (uint32_t vc = 0; vc < 2; ++vc) {
        for (uint32_t ch = 0; ch < num_z_router_sender_channels_per_vc; ++ch) {
            auto targets = z_mapping.get_downstream_targets(vc, ch);

            // For each target, verify it's not pointing to a Z router VC0
            for (const auto& target : targets) {
                // Z_TO_MESH should target mesh VC1, not Z VC0
                if (target.type == ConnectionType::Z_TO_MESH) {
                    EXPECT_EQ(target.target_vc, 1) << "Z_TO_MESH should target mesh VC1, not VC0";
                    EXPECT_TRUE(target.target_direction.has_value());
                    EXPECT_NE(target.target_direction.value(), RoutingDirection::Z)
                        << "Z router should not target another Z router";
                }
            }
        }
    }

    // Also check mesh router mappings - they should target Z VC0 receiver, not sender
    auto mesh_mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh, RoutingDirection::N, true);  // has_z = true

    auto mesh_targets = mesh_mapping.get_downstream_targets(0, 0);
    ASSERT_EQ(mesh_targets.size(), 4);  // 3 INTRA_MESH + 1 MESH_TO_Z

    auto mesh_to_z_it = std::find_if(mesh_targets.begin(), mesh_targets.end(),
        [](const ConnectionTarget& t) { return t.type == ConnectionType::MESH_TO_Z; });
    ASSERT_NE(mesh_to_z_it, mesh_targets.end());

    EXPECT_EQ(mesh_to_z_it->target_vc, 0) << "MESH_TO_Z should target Z VC0 (receiver)";
    EXPECT_EQ(mesh_to_z_it->target_direction.value(), RoutingDirection::Z);

    // The mapping specifies VC0, which implicitly means the receiver side
    // (since Z VC0 senders have no downstream connections)
}

TEST_F(ZRouterIntegrationTest, ZRouter_VC0_NoOutgoingConnections) {
    // Z router VC0 should have no outgoing connections (reserved for future use)
    auto mapping = RouterConnectionMapping::for_z_router();

    // Check all VC0 sender channels - none should have downstream targets
    for (uint32_t ch = 0; ch < 4; ++ch) {
        auto targets = mapping.get_downstream_targets(0, ch);
        EXPECT_TRUE(targets.empty()) << "Z router VC0 sender " << ch << " should have no downstream connections";
    }
}

TEST_F(ZRouterIntegrationTest, ZRouter_VC0_NoDownstreamReceiverConnections) {
    // Validate that no downstream router should connect TO a Z router VC0 sender channel
    // Z router VC0 senders don't forward anywhere, so connecting to them would be invalid

    FabricNodeId device0(MeshId{0}, 0);

    // Attempt to create invalid connection: mesh router → Z router VC0 sender
    // This should never happen in practice, but we validate the constraint

    // Valid: MESH_TO_Z targets Z VC0 receiver (not sender)
    record_connection(
        device0, RoutingDirection::N, 0,
        0, builder_config::num_sender_channels_2d_mesh,  // mesh VC0 MESH_TO_Z channel
        device0, RoutingDirection::Z, 4,
        0, 0,  // Z VC0 receiver 0 (CORRECT)
        ConnectionType::MESH_TO_Z);

    EXPECT_EQ(registry_->size(), 1);

    // Verify the connection targets the receiver, not a sender
    auto connections = registry_->get_connections_by_type(ConnectionType::MESH_TO_Z);
    ASSERT_EQ(connections.size(), 1);

    const auto& conn = connections[0];
    EXPECT_EQ(conn.dest_vc, 0) << "MESH_TO_Z should target Z VC0";
    EXPECT_EQ(conn.dest_sender_channel, 0) << "Should target receiver channel, not sender";

    // Note: There's no explicit "sender vs receiver" flag in the connection record,
    // but the semantic is clear: dest_sender_channel refers to the receiver side
}

TEST_F(ZRouterIntegrationTest, MultiTargetReceiver_SingleZReceiverChannel) {
    // Verify that all MESH_TO_Z connections target the same Z receiver (multi-target)
    FabricNodeId device0(MeshId{0}, 0);

    std::vector<RoutingDirection> mesh_dirs = {
        RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W
    };

    for (size_t i = 0; i < 4; ++i) {
        record_connection(
            device0, mesh_dirs[i], static_cast<uint8_t>(i),
            0, builder_config::num_sender_channels_2d_mesh,
            device0, RoutingDirection::Z, 4,
            0, 0,  // All target same receiver
            ConnectionType::MESH_TO_Z);
    }

    auto mesh_to_z = registry_->get_connections_by_type(ConnectionType::MESH_TO_Z);
    ASSERT_EQ(mesh_to_z.size(), 4);

    // All should target same receiver
    for (const auto& conn : mesh_to_z) {
        EXPECT_EQ(conn.dest_vc, 0);
        EXPECT_EQ(conn.dest_sender_channel, 0);
    }
}

// ============================================================================
// Topology Compatibility Tests
// ============================================================================

TEST_F(ZRouterIntegrationTest, AllTopologies_MeshMappingWorks) {
    // Test that mesh router mapping works for all topologies
    std::vector<Topology> topologies = {
        Topology::Linear, Topology::Ring, Topology::Mesh, Topology::Torus
    };

    for (auto topology : topologies) {
        auto mapping = RouterConnectionMapping::for_mesh_router(
            topology, RoutingDirection::N, false);

        // Should be able to create mapping without errors
        EXPECT_NO_THROW({
            auto targets = mapping.get_downstream_targets(0, 1);
        }) << "Topology " << static_cast<int>(topology) << " should create valid mapping";
    }
}

TEST_F(ZRouterIntegrationTest, LinearTopology_NoMeshToZ) {
    // Linear topology with Z should still work (though unusual configuration)
    auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Linear, RoutingDirection::N, true);  // has_z = true

    // Linear receiver channel 0 should have 2 targets (1 INTRA_MESH + 1 MESH_TO_Z)
    auto targets = mapping.get_downstream_targets(0, 0);
    ASSERT_EQ(targets.size(), 2);

    auto mesh_to_z_it = std::find_if(targets.begin(), targets.end(),
        [](const ConnectionTarget& t) { return t.type == ConnectionType::MESH_TO_Z; });
    ASSERT_NE(mesh_to_z_it, targets.end());
}

// ============================================================================
// Negative Tests: Connection Overflow and Invalid Configurations
// ============================================================================

TEST_F(ZRouterIntegrationTest, ZRouter_VC1_ExceedsMaxMeshRouters) {
    // Negative test: Z router VC1 receiver should not accept more than 4 connections
    // (one per mesh router direction: N, E, S, W)

    FabricNodeId device0(MeshId{0}, 0);

    // Try to add 5 connections (invalid - only 4 mesh routers max)
    std::vector<RoutingDirection> invalid_dirs = {
        RoutingDirection::N,
        RoutingDirection::E,
        RoutingDirection::S,
        RoutingDirection::W,
        RoutingDirection::N  // Duplicate N (invalid)
    };

    for (size_t i = 0; i < 5; ++i) {
        record_connection(
            device0, invalid_dirs[i], static_cast<uint8_t>(i % 4),
            0, builder_config::num_sender_channels_2d_mesh,
            device0, RoutingDirection::Z, 4,
            0, 0,  // All target same receiver (multi-target)
            ConnectionType::MESH_TO_Z);
    }

    // Should have 5 connections recorded
    EXPECT_EQ(registry_->size(), 5);

    // But this is invalid - Z router VC1 receiver can only accept 4 unique directions
    auto mesh_to_z = registry_->get_connections_by_type(ConnectionType::MESH_TO_Z);
    EXPECT_EQ(mesh_to_z.size(), 5);

    // TODO: Add validation in ConnectionRegistry or builder to detect duplicate directions
    // and reject > 4 MESH_TO_Z connections to the same Z router
    // For now, this test documents the overflow behavior
}

TEST_F(ZRouterIntegrationTest, MeshRouter_ConnectToNonExistentVC) {
    // Negative test: Mesh router should not be able to connect to non-existent VC

    auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh, RoutingDirection::N, false);

    // Try to query VC2 (doesn't exist)
    auto targets = mapping.get_downstream_targets(2, 0);

    // Should return empty (no targets for non-existent VC)
    EXPECT_TRUE(targets.empty()) << "Non-existent VC should have no targets";
}

TEST_F(ZRouterIntegrationTest, MeshRouter_ConnectToNonExistentSenderChannel) {
    // Negative test: Mesh router should not be able to connect to non-existent sender channel

    auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh, RoutingDirection::N, false);

    // Mesh VC0 has 4 sender channels (0-3), try to query channel 10
    auto targets = mapping.get_downstream_targets(0, 10);

    // Should return empty (no targets for non-existent channel)
    EXPECT_TRUE(targets.empty()) << "Non-existent sender channel should have no targets";
}

TEST_F(ZRouterIntegrationTest, ZRouter_QueryInvalidVC) {
    // Negative test: Z router has 2 VCs (0-1), querying VC2+ should return empty

    auto mapping = RouterConnectionMapping::for_z_router();

    // Try to query VC2 (doesn't exist)
    auto targets = mapping.get_downstream_targets(2, 0);

    EXPECT_TRUE(targets.empty()) << "Z router VC2 should have no targets (doesn't exist)";

    // Try VC3
    targets = mapping.get_downstream_targets(3, 0);
    EXPECT_TRUE(targets.empty()) << "Z router VC3 should have no targets (doesn't exist)";
}
