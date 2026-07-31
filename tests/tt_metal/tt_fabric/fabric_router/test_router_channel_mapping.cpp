// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "tt_metal/fabric/fabric_router_channel_mapping.hpp"
#include "tt_metal/fabric/builder/fabric_edge_capability.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <hostdevcommon/fabric_common.h>

namespace tt::tt_fabric {

/**
 * FabricRouterChannelMapping Tests
 *
 * The per-VC channel shape and flat layout of one router, all of it read off router_vc_shape()
 * in builder/router_wiring_rules.*:
 *
 * - Legacy mesh/1D: VC0 only (4 senders in 2D, 2 in 1D); VC1 appears with an intermesh config.
 * - The intermesh boundary family: VC0 = worker + 4 wired producers (5), VC1 = the 4-wide
 *   from-boundary fanout; VC1 senders start at flat index 5.
 * - Express: VC0 widens to the family max of 5, VC1 to 4; VC1 senders start after the wide base.
 * - VC2: one sender at the flat base of the family (7 legacy, 8 on a boundary chip, 9 express),
 *   and no VC2 receiver on the boundary router.
 * - IntermeshVCConfig factory flags and invalid-query failures.
 *
 * Turn sets (what those senders connect to) live in test_router_connection_mapping.cpp.
 */

class RouterChannelMappingTest : public ::testing::Test {};

// ============ Legacy mesh / 1D ============

TEST_F(RouterChannelMappingTest, LegacyMesh_AndTorus_VC0OnlyLayout) {
    for (auto topology : {Topology::Mesh, Topology::Torus}) {
        FabricRouterChannelMapping mapping(
            topology, false, RoutingDirection::N, EdgeCapability::INTRAMESH_CARDINAL, nullptr);

        EXPECT_EQ(mapping.get_num_virtual_channels(), 1) << "topology " << static_cast<int>(topology);
        EXPECT_EQ(mapping.get_num_sender_channels_for_vc(0), 4);
        EXPECT_EQ(mapping.get_num_sender_channels_for_vc(1), 0);

        for (uint32_t i = 0; i < 4; ++i) {
            auto sender = mapping.get_sender_mapping(0, i);
            EXPECT_EQ(sender.builder_type, BuilderType::ERISC);
            EXPECT_EQ(sender.internal_sender_channel_id, i);
        }
    }
}

TEST_F(RouterChannelMappingTest, Linear_VC0OnlyLayout) {
    FabricRouterChannelMapping mapping(
        Topology::Linear, false, RoutingDirection::N, EdgeCapability::INTRAMESH_CARDINAL, nullptr);

    EXPECT_EQ(mapping.get_num_virtual_channels(), 1);
    EXPECT_EQ(mapping.get_num_sender_channels_for_vc(0), 2);
    EXPECT_EQ(mapping.get_num_sender_channels_for_vc(1), 0);

    for (uint32_t i = 0; i < 2; ++i) {
        auto sender = mapping.get_sender_mapping(0, i);
        EXPECT_EQ(sender.builder_type, BuilderType::ERISC);
        EXPECT_EQ(sender.internal_sender_channel_id, i);
    }
}

TEST_F(RouterChannelMappingTest, TensixExtension_VC0SendersMapToTensix) {
    FabricRouterChannelMapping mapping(
        Topology::Mesh, true, RoutingDirection::N, EdgeCapability::INTRAMESH_CARDINAL, nullptr);

    for (uint32_t i = 0; i < 4; ++i) {
        auto sender = mapping.get_sender_mapping(0, i);
        EXPECT_EQ(sender.builder_type, BuilderType::TENSIX);
        EXPECT_EQ(sender.internal_sender_channel_id, i);
    }
    EXPECT_EQ(mapping.get_receiver_mapping(0, 0).builder_type, BuilderType::ERISC);
}

// ============ VC1 on ordinary mesh routers (intermesh config) ============

TEST_F(RouterChannelMappingTest, MeshRouter_IntermeshVC1Layout) {
    // Any intermesh mode enables VC1; the mode does not change the channel layout.
    for (const auto& config :
         {IntermeshVCConfig::edge_only(),
          IntermeshVCConfig::full_mesh(),
          IntermeshVCConfig::full_mesh_with_pass_through()}) {
        FabricRouterChannelMapping mapping(
            Topology::Mesh, false, RoutingDirection::N, EdgeCapability::INTRAMESH_CARDINAL, &config);

        EXPECT_EQ(mapping.get_num_virtual_channels(), 2);
        EXPECT_EQ(mapping.get_num_sender_channels_for_vc(1), 3) << "3 cardinal producers, no extra-port slot";
        // VC1 senders are laid out immediately after the four VC0 senders.
        for (uint32_t i = 0; i < 3; ++i) {
            auto sender = mapping.get_sender_mapping(1, i);
            EXPECT_EQ(sender.builder_type, BuilderType::ERISC);
            EXPECT_EQ(sender.internal_sender_channel_id, 4 + i);
        }
    }
}

TEST_F(RouterChannelMappingTest, BoundaryChipMeshRouter_VC1HasExtraFromBoundarySlot) {
    // On a chip whose extra port is an intermesh boundary, VC1 widens by the from-boundary slot.
    auto config = IntermeshVCConfig::full_mesh();
    FabricRouterChannelMapping mapping(
        Topology::Mesh,
        false,
        RoutingDirection::N,
        EdgeCapability::INTRAMESH_CARDINAL,
        &config,
        ZPortRole::INTERMESH_BOUNDARY);

    EXPECT_EQ(mapping.get_num_virtual_channels(), 2);
    EXPECT_EQ(mapping.get_num_sender_channels_for_vc(1), 4) << "3 cardinals + the from-boundary producer";
    for (uint32_t i = 0; i < 4; ++i) {
        EXPECT_EQ(mapping.get_sender_mapping(1, i).internal_sender_channel_id, 4 + i);
    }
}

// ============ The intermesh boundary router ============

TEST_F(RouterChannelMappingTest, IntermeshBoundaryRouter_CompleteLayout) {
    auto config = IntermeshVCConfig::full_mesh();
    FabricRouterChannelMapping mapping(
        Topology::Mesh, false, RoutingDirection::Z, EdgeCapability::INTERMESH, &config, ZPortRole::INTERMESH_BOUNDARY);

    EXPECT_EQ(mapping.get_num_virtual_channels(), 2);

    // VC0: worker + one slot per wired mesh producer.
    EXPECT_EQ(mapping.get_num_sender_channels_for_vc(0), 5);
    for (uint32_t i = 0; i < 5; ++i) {
        auto sender = mapping.get_sender_mapping(0, i);
        EXPECT_EQ(sender.builder_type, BuilderType::ERISC);
        EXPECT_EQ(sender.internal_sender_channel_id, i);
    }

    // VC1: the four-wide from-boundary fanout, starting after the five-wide VC0.
    EXPECT_EQ(mapping.get_num_sender_channels_for_vc(1), 4);
    for (uint32_t i = 0; i < 4; ++i) {
        auto sender = mapping.get_sender_mapping(1, i);
        EXPECT_EQ(sender.builder_type, BuilderType::ERISC);
        EXPECT_EQ(sender.internal_sender_channel_id, 5 + i);
    }

    // One receiver per VC, in VC order.
    EXPECT_EQ(mapping.get_receiver_mapping(0, 0).internal_receiver_channel_id, 0);
    EXPECT_EQ(mapping.get_receiver_mapping(1, 0).internal_receiver_channel_id, 1);

    // Flat enumeration is VC0 then VC1, in order.
    const auto all = mapping.get_all_sender_mappings();
    ASSERT_EQ(all.size(), 9);
    for (size_t i = 0; i < all.size(); ++i) {
        EXPECT_EQ(all[i].internal_sender_channel_id, i);
    }
}

// ============ Express ============

TEST_F(RouterChannelMappingTest, ExpressMesh_WidenedVC0AndVC1Base) {
    // Express widens VC0 to the family max (worker + 3 cardinals + the chord slot) and VC1 to
    // the four wired producers; VC1 senders are laid out after the five-wide VC0, not the 2D
    // mesh constant -- otherwise VC1 sender 0 would alias VC0's express channel at flat index 4.
    auto config = IntermeshVCConfig::full_mesh();
    FabricRouterChannelMapping mapping(
        Topology::Torus,
        false,
        RoutingDirection::N,
        EdgeCapability::INTRAMESH_CARDINAL,
        &config,
        ZPortRole::NONE,
        /*express_routing_enabled=*/true);

    EXPECT_EQ(mapping.get_num_virtual_channels(), 2);
    EXPECT_EQ(mapping.get_num_sender_channels_for_vc(0), 5);
    EXPECT_EQ(mapping.get_num_sender_channels_for_vc(1), 4);
    for (uint32_t i = 0; i < 4; ++i) {
        auto sender = mapping.get_sender_mapping(1, i);
        EXPECT_EQ(sender.builder_type, BuilderType::ERISC);
        EXPECT_EQ(sender.internal_sender_channel_id, 5 + i);
    }
}

TEST_F(RouterChannelMappingTest, ExpressMesh_VC2AtFlatIndex9) {
    // 5 + 4 + 1 reaches the num_max_sender_channels ceiling exactly.
    auto config = IntermeshVCConfig::full_mesh();
    config.requires_vc2 = true;
    FabricRouterChannelMapping mapping(
        Topology::Torus,
        false,
        RoutingDirection::N,
        EdgeCapability::INTRAMESH_CARDINAL,
        &config,
        ZPortRole::NONE,
        /*express_routing_enabled=*/true);

    EXPECT_EQ(mapping.get_num_virtual_channels(), 3);
    auto vc2_sender = mapping.get_sender_mapping(2, 0);
    EXPECT_EQ(vc2_sender.builder_type, BuilderType::ERISC);
    EXPECT_EQ(vc2_sender.internal_sender_channel_id, 9);
}

// ============ VC2 on legacy and boundary families ============

TEST_F(RouterChannelMappingTest, VC2_LegacyMeshFlatIndices) {
    auto config = IntermeshVCConfig::full_mesh();
    config.requires_vc2 = true;
    FabricRouterChannelMapping mapping(
        Topology::Mesh, false, RoutingDirection::N, EdgeCapability::INTRAMESH_CARDINAL, &config, ZPortRole::NONE);

    EXPECT_EQ(mapping.get_num_virtual_channels(), 3);
    EXPECT_EQ(mapping.get_num_sender_channels_for_vc(2), 1);
    // VC0:4 + VC1:3 places the VC2 sender at flat index 7; the receiver follows VC0/VC1.
    EXPECT_EQ(mapping.get_sender_mapping(2, 0).internal_sender_channel_id, 7);
    EXPECT_EQ(mapping.get_receiver_mapping(2, 0).internal_receiver_channel_id, 2);

    // The existing VCs are untouched by the extra sender.
    for (uint32_t i = 0; i < 4; ++i) {
        EXPECT_EQ(mapping.get_sender_mapping(0, i).internal_sender_channel_id, i);
    }
    for (uint32_t i = 0; i < 3; ++i) {
        EXPECT_EQ(mapping.get_sender_mapping(1, i).internal_sender_channel_id, 4 + i);
    }
}

TEST_F(RouterChannelMappingTest, VC2_BoundaryChipFlatIndices) {
    // A boundary-chip mesh router: VC0:4 + VC1:4 (the extra from-boundary slot) places the VC2
    // sender at flat index 8.
    auto config = IntermeshVCConfig::full_mesh();
    config.requires_vc2 = true;
    FabricRouterChannelMapping mapping(
        Topology::Mesh,
        false,
        RoutingDirection::N,
        EdgeCapability::INTRAMESH_CARDINAL,
        &config,
        ZPortRole::INTERMESH_BOUNDARY);

    EXPECT_EQ(mapping.get_num_virtual_channels(), 3);
    EXPECT_EQ(mapping.get_sender_mapping(2, 0).internal_sender_channel_id, 8);
}

TEST_F(RouterChannelMappingTest, VC2_BoundaryRouterFlatIndexAndNoReceiver) {
    // The boundary router: VC0:5 + VC1:4 places the VC2 sender at flat index 9, and the boundary
    // services no VC2 receiver.
    auto config = IntermeshVCConfig::full_mesh();
    config.requires_vc2 = true;
    FabricRouterChannelMapping mapping(
        Topology::Mesh, false, RoutingDirection::Z, EdgeCapability::INTERMESH, &config, ZPortRole::INTERMESH_BOUNDARY);

    EXPECT_EQ(mapping.get_num_virtual_channels(), 3);
    EXPECT_EQ(mapping.get_num_sender_channels_for_vc(2), 1);
    EXPECT_EQ(mapping.get_sender_mapping(2, 0).internal_sender_channel_id, 9);
    EXPECT_THROW(mapping.get_receiver_mapping(2, 0), std::exception);
}

TEST_F(RouterChannelMappingTest, VC2_RequiresExplicitEnable) {
    auto config = IntermeshVCConfig::full_mesh();  // does not set requires_vc2
    FabricRouterChannelMapping mapping(
        Topology::Mesh, false, RoutingDirection::N, EdgeCapability::INTRAMESH_CARDINAL, &config);

    EXPECT_EQ(mapping.get_num_virtual_channels(), 2);
    EXPECT_EQ(mapping.get_num_sender_channels_for_vc(2), 0);
}

// ============ IntermeshVCConfig factories ============

TEST_F(RouterChannelMappingTest, IntermeshVCConfig_FactoryFlags) {
    struct Expected {
        bool vc1, full_mesh, pass_through;
    };
    const std::vector<std::pair<IntermeshVCConfig, Expected>> cases = {
        {IntermeshVCConfig::disabled(), {false, false, false}},
        {IntermeshVCConfig::edge_only(), {true, false, false}},
        {IntermeshVCConfig::full_mesh(), {true, true, false}},
        {IntermeshVCConfig::full_mesh_with_pass_through(), {true, true, true}},
        {IntermeshVCConfig{}, {false, false, false}},  // default-constructed is DISABLED
    };

    for (const auto& [config, expected] : cases) {
        EXPECT_EQ(config.requires_vc1, expected.vc1) << "mode " << static_cast<int>(config.mode);
        EXPECT_EQ(config.requires_vc1_full_mesh, expected.full_mesh) << "mode " << static_cast<int>(config.mode);
        EXPECT_EQ(config.requires_vc1_mesh_pass_through, expected.pass_through)
            << "mode " << static_cast<int>(config.mode);
    }
}

// ============ Invalid queries ============

TEST_F(RouterChannelMappingTest, InvalidQueries_Throw) {
    FabricRouterChannelMapping mesh(
        Topology::Mesh, false, RoutingDirection::N, EdgeCapability::INTRAMESH_CARDINAL, nullptr);
    EXPECT_THROW(mesh.get_sender_mapping(5, 0), std::exception);
    EXPECT_THROW(mesh.get_sender_mapping(0, 10), std::exception);

    auto config = IntermeshVCConfig::full_mesh();
    FabricRouterChannelMapping boundary(
        Topology::Mesh, false, RoutingDirection::Z, EdgeCapability::INTERMESH, &config, ZPortRole::INTERMESH_BOUNDARY);
    EXPECT_THROW(boundary.get_receiver_mapping(1, 5), std::exception);
}

}  // namespace tt::tt_fabric
