// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <enchantum/enchantum.hpp>
#include "tt_metal/fabric/builder/router_wiring_rules.hpp"
#include "tt_metal/fabric/builder/fabric_edge_capability.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <hostdevcommon/fabric_common.h>

namespace tt::tt_fabric {

/**
 * RouterVcShape / layout Tests
 *
 * The per-VC channel shape of one router and the flat layout read off it, from router_vc_shape()
 * and the shape's flat_sender_id/flat_receiver_id prefix sums:
 *
 * - Non-express mesh/1D: VC0 only (4 senders in 2D, 2 in 1D); VC1 appears with an intermesh config.
 * - The intermesh boundary family: VC0 = worker + 4 wired producers (5), VC1 = the 4-wide
 *   from-boundary fanout; VC1 senders start at flat index 5.
 * - Express: VC0 widens to the family max of 5, VC1 to 4; VC1 senders start after the wide base.
 * - VC2: one sender at the flat base of the family (7 non-express, 8 on a boundary chip, 9 express),
 *   and no VC2 receiver on the boundary router.
 * - Builder ownership (builder_type_for_vc): tensix takes VC0 in MUX mode, nothing else.
 * - IntermeshVCConfig factory flags and invalid-query failures.
 *
 * The wiring primitive itself is pinned in test_express_connection_wiring.cpp, and the turn sets
 * built from it in test_router_turn_set.cpp.
 */

class RouterWiringRulesTest : public ::testing::Test {};

namespace {

// One chip, named by the only fact that distinguishes the families here: what its Z port is for.
// Every cardinal is an ordinary same-mesh edge, so the router's own capability and the chip's
// Z-port role are both read back off the set rather than passed beside it.
PerDirectionCapabilities chip_with_z(std::optional<EdgeCapability> z_capability) {
    PerDirectionCapabilities caps;
    for (const auto direction : {RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W}) {
        caps.at(direction) = EdgeCapability::INTRAMESH_CARDINAL;
    }
    caps.at(RoutingDirection::Z) = z_capability;
    return caps;
}

}  // namespace

// ============ Non-express mesh / 1D ============

TEST_F(RouterWiringRulesTest, NonExpressMesh_AndTorus_VC0OnlyLayout) {
    for (auto topology : {Topology::Mesh, Topology::Torus}) {
        const auto shape = router_vc_shape(
            topology,
            RoutingDirection::N,
            chip_with_z(std::nullopt),
            /*express_routing_enabled=*/false,
            nullptr);

        EXPECT_EQ(shape.num_vcs, 1) << "topology " << enchantum::to_string(topology);
        EXPECT_EQ(shape.sender_counts[0], 4);
        EXPECT_EQ(shape.sender_counts[1], 0);

        for (uint32_t i = 0; i < 4; ++i) {
            EXPECT_EQ(shape.flat_sender_id(0, i), i);
        }
    }
}

TEST_F(RouterWiringRulesTest, Linear_VC0OnlyLayout) {
    const auto shape = router_vc_shape(
        Topology::Linear,
        RoutingDirection::N,
        chip_with_z(std::nullopt),
        /*express_routing_enabled=*/false,
        nullptr);

    EXPECT_EQ(shape.num_vcs, 1);
    EXPECT_EQ(shape.sender_counts[0], 2);
    EXPECT_EQ(shape.sender_counts[1], 0);

    for (uint32_t i = 0; i < 2; ++i) {
        EXPECT_EQ(shape.flat_sender_id(0, i), i);
    }
}

TEST_F(RouterWiringRulesTest, BuilderOwnership_TensixTakesVC0Only) {
    EXPECT_EQ(builder_type_for_vc(0, /*downstream_is_tensix_builder=*/true), BuilderType::TENSIX);
    EXPECT_EQ(builder_type_for_vc(1, true), BuilderType::ERISC);
    EXPECT_EQ(builder_type_for_vc(2, true), BuilderType::ERISC);
    EXPECT_EQ(builder_type_for_vc(0, false), BuilderType::ERISC);
}

// ============ VC1 on ordinary mesh routers (intermesh config) ============

TEST_F(RouterWiringRulesTest, MeshRouter_IntermeshVC1Layout) {
    // Any intermesh mode enables VC1; the mode does not change the channel layout.
    for (const auto& config :
         {IntermeshVCConfig::edge_only(),
          IntermeshVCConfig::full_mesh(),
          IntermeshVCConfig::full_mesh_with_pass_through()}) {
        const auto shape = router_vc_shape(
            Topology::Mesh,
            RoutingDirection::N,
            chip_with_z(std::nullopt),
            /*express_routing_enabled=*/false,
            &config);

        EXPECT_EQ(shape.num_vcs, 2);
        EXPECT_EQ(shape.sender_counts[1], 3) << "3 cardinal producers, no extra-port slot";
        // VC1 senders are laid out immediately after the four VC0 senders.
        for (uint32_t i = 0; i < 3; ++i) {
            EXPECT_EQ(shape.flat_sender_id(1, i), 4 + i);
        }
    }
}

TEST_F(RouterWiringRulesTest, BoundaryChipMeshRouter_VC1HasExtraFromBoundarySlot) {
    // On a chip whose extra port is an intermesh boundary, VC1 widens by the from-boundary slot.
    auto config = IntermeshVCConfig::full_mesh();
    const auto shape = router_vc_shape(
        Topology::Mesh,
        RoutingDirection::N,
        chip_with_z(EdgeCapability::INTERMESH),
        /*express_routing_enabled=*/false,
        &config);

    EXPECT_EQ(shape.num_vcs, 2);
    EXPECT_EQ(shape.sender_counts[1], 4) << "3 cardinals + the from-boundary producer";
    for (uint32_t i = 0; i < 4; ++i) {
        EXPECT_EQ(shape.flat_sender_id(1, i), 4 + i);
    }
}

// ============ The intermesh boundary router ============

TEST_F(RouterWiringRulesTest, IntermeshBoundaryRouter_CompleteLayout) {
    auto config = IntermeshVCConfig::full_mesh();
    const auto shape = router_vc_shape(
        Topology::Mesh,
        RoutingDirection::Z,
        chip_with_z(EdgeCapability::INTERMESH),
        /*express_routing_enabled=*/false,
        &config);

    EXPECT_EQ(shape.num_vcs, 2);

    // VC0: worker + one slot per wired mesh producer.
    EXPECT_EQ(shape.sender_counts[0], 5);
    for (uint32_t i = 0; i < 5; ++i) {
        EXPECT_EQ(shape.flat_sender_id(0, i), i);
    }

    // VC1: the four-wide from-boundary fanout, starting after the five-wide VC0.
    EXPECT_EQ(shape.sender_counts[1], 4);
    for (uint32_t i = 0; i < 4; ++i) {
        EXPECT_EQ(shape.flat_sender_id(1, i), 5 + i);
    }

    // One receiver per VC, in VC order.
    EXPECT_EQ(shape.flat_receiver_id(0, 0), 0);
    EXPECT_EQ(shape.flat_receiver_id(1, 0), 1);

    // The flat sender enumeration is VC0 then VC1 with no gaps: 0..8.
    uint32_t expected = 0;
    for (uint32_t vc = 0; vc < shape.num_vcs; ++vc) {
        for (uint32_t ch = 0; ch < shape.sender_counts[vc]; ++ch) {
            EXPECT_EQ(shape.flat_sender_id(vc, ch), expected++);
        }
    }
    EXPECT_EQ(expected, 9);
}

// ============ Express ============

TEST_F(RouterWiringRulesTest, ExpressMesh_WidenedVC0AndVC1Base) {
    // Express widens VC0 to the family max (worker + 3 cardinals + the chord slot) and VC1 to
    // the four wired producers; VC1 senders are laid out after the five-wide VC0, not the 2D
    // mesh constant -- otherwise VC1 sender 0 would alias VC0's express channel at flat index 4.
    auto config = IntermeshVCConfig::full_mesh();
    const auto shape = router_vc_shape(
        Topology::Torus,
        RoutingDirection::N,
        chip_with_z(std::nullopt),
        /*express_routing_enabled=*/true,
        &config);

    EXPECT_EQ(shape.num_vcs, 2);
    EXPECT_EQ(shape.sender_counts[0], 5);
    EXPECT_EQ(shape.sender_counts[1], 4);
    for (uint32_t i = 0; i < 4; ++i) {
        EXPECT_EQ(shape.flat_sender_id(1, i), 5 + i);
    }
}

TEST_F(RouterWiringRulesTest, ExpressMesh_VC2AtFlatIndex9) {
    // 5 + 4 + 1 reaches the num_max_sender_channels ceiling exactly.
    auto config = IntermeshVCConfig::full_mesh();
    config.requires_vc2 = true;
    const auto shape = router_vc_shape(
        Topology::Torus,
        RoutingDirection::N,
        chip_with_z(std::nullopt),
        /*express_routing_enabled=*/true,
        &config);

    EXPECT_EQ(shape.num_vcs, 3);
    EXPECT_EQ(shape.flat_sender_id(2, 0), 9);
}

// ============ VC2 on non-express and boundary families ============

TEST_F(RouterWiringRulesTest, VC2_NonExpressMeshFlatIndices) {
    auto config = IntermeshVCConfig::full_mesh();
    config.requires_vc2 = true;
    const auto shape = router_vc_shape(
        Topology::Mesh,
        RoutingDirection::N,
        chip_with_z(std::nullopt),
        /*express_routing_enabled=*/false,
        &config);

    EXPECT_EQ(shape.num_vcs, 3);
    EXPECT_EQ(shape.sender_counts[2], 1);
    // VC0:4 + VC1:3 places the VC2 sender at flat index 7; the receiver follows VC0/VC1.
    EXPECT_EQ(shape.flat_sender_id(2, 0), 7);
    EXPECT_EQ(shape.flat_receiver_id(2, 0), 2);

    // The existing VCs are untouched by the extra sender.
    for (uint32_t i = 0; i < 4; ++i) {
        EXPECT_EQ(shape.flat_sender_id(0, i), i);
    }
    for (uint32_t i = 0; i < 3; ++i) {
        EXPECT_EQ(shape.flat_sender_id(1, i), 4 + i);
    }
}

TEST_F(RouterWiringRulesTest, VC2_BoundaryChipFlatIndices) {
    // A boundary-chip mesh router: VC0:4 + VC1:4 (the extra from-boundary slot) places the VC2
    // sender at flat index 8.
    auto config = IntermeshVCConfig::full_mesh();
    config.requires_vc2 = true;
    const auto shape = router_vc_shape(
        Topology::Mesh,
        RoutingDirection::N,
        chip_with_z(EdgeCapability::INTERMESH),
        /*express_routing_enabled=*/false,
        &config);

    EXPECT_EQ(shape.num_vcs, 3);
    EXPECT_EQ(shape.flat_sender_id(2, 0), 8);
}

TEST_F(RouterWiringRulesTest, VC2_BoundaryRouterFlatIndexAndNoReceiver) {
    // The boundary router: VC0:5 + VC1:4 places the VC2 sender at flat index 9, and the boundary
    // services no VC2 receiver.
    auto config = IntermeshVCConfig::full_mesh();
    config.requires_vc2 = true;
    const auto shape = router_vc_shape(
        Topology::Mesh,
        RoutingDirection::Z,
        chip_with_z(EdgeCapability::INTERMESH),
        /*express_routing_enabled=*/false,
        &config);

    EXPECT_EQ(shape.num_vcs, 3);
    EXPECT_EQ(shape.sender_counts[2], 1);
    EXPECT_EQ(shape.flat_sender_id(2, 0), 9);
    EXPECT_EQ(shape.receiver_counts[2], 0);
    EXPECT_THROW((void)shape.flat_receiver_id(2, 0), std::exception);
}

TEST_F(RouterWiringRulesTest, VC2_RequiresExplicitEnable) {
    auto config = IntermeshVCConfig::full_mesh();  // does not set requires_vc2
    const auto shape = router_vc_shape(
        Topology::Mesh,
        RoutingDirection::N,
        chip_with_z(std::nullopt),
        /*express_routing_enabled=*/false,
        &config);

    EXPECT_EQ(shape.num_vcs, 2);
    EXPECT_EQ(shape.sender_counts[2], 0);
}

// ============ IntermeshVCConfig factories ============

TEST_F(RouterWiringRulesTest, IntermeshVCConfig_FactoryFlags) {
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
        EXPECT_EQ(config.requires_vc1, expected.vc1) << "mode " << enchantum::to_string(config.mode);
        EXPECT_EQ(config.requires_vc1_full_mesh, expected.full_mesh) << "mode " << enchantum::to_string(config.mode);
        EXPECT_EQ(config.requires_vc1_mesh_pass_through, expected.pass_through)
            << "mode " << enchantum::to_string(config.mode);
    }
}

// ============ Invalid queries ============

TEST_F(RouterWiringRulesTest, InvalidQueries_Throw) {
    const auto mesh = router_vc_shape(
        Topology::Mesh,
        RoutingDirection::N,
        chip_with_z(std::nullopt),
        /*express_routing_enabled=*/false,
        nullptr);
    EXPECT_THROW((void)mesh.flat_sender_id(5, 0), std::exception);
    EXPECT_THROW((void)mesh.flat_sender_id(0, 10), std::exception);

    auto config = IntermeshVCConfig::full_mesh();
    const auto boundary = router_vc_shape(
        Topology::Mesh,
        RoutingDirection::Z,
        chip_with_z(EdgeCapability::INTERMESH),
        /*express_routing_enabled=*/false,
        &config);
    EXPECT_THROW((void)boundary.flat_receiver_id(1, 5), std::exception);
}

}  // namespace tt::tt_fabric
