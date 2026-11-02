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

// RouterVcShape family counts and flat-layout boundaries.

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

TEST_F(RouterWiringRulesTest, NonExpress2D_VC0OnlyLayout) {
    const auto shape = router_vc_shape(Topology::Mesh, RoutingDirection::N, chip_with_z(std::nullopt), false, nullptr);
    EXPECT_EQ(shape.num_vcs, 1);
    EXPECT_EQ(shape.sender_counts[0], 4);
    EXPECT_EQ(shape.sender_counts[1], 0);
    EXPECT_EQ(shape.flat_sender_id(0, 3), 3);
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
    EXPECT_EQ(shape.flat_sender_id(0, 1), 1);
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
        EXPECT_EQ(shape.sender_counts[2], 0);
        EXPECT_EQ(shape.flat_sender_id(1, 0), 4);
        EXPECT_EQ(shape.flat_sender_id(1, 2), 6);
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
    EXPECT_EQ(shape.flat_sender_id(1, 0), 4);
    EXPECT_EQ(shape.flat_sender_id(1, 3), 7);
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
    EXPECT_EQ(shape.flat_sender_id(1, 0), 5);
    EXPECT_EQ(shape.flat_sender_id(1, 3), 8);
}

TEST_F(RouterWiringRulesTest, VC2FlatLayoutByRouterFamily) {
    struct Case {
        Topology topology;
        RoutingDirection facing;
        std::optional<EdgeCapability> z_capability;
        bool express;
        uint32_t expected_sender;
        bool has_receiver;
    };
    const std::vector<Case> cases = {
        {Topology::Torus, RoutingDirection::N, std::nullopt, true, 9, true},
        {Topology::Mesh, RoutingDirection::N, std::nullopt, false, 7, true},
        {Topology::Mesh, RoutingDirection::N, EdgeCapability::INTERMESH, false, 8, true},
        {Topology::Mesh, RoutingDirection::Z, EdgeCapability::INTERMESH, false, 9, false},
    };

    auto config = IntermeshVCConfig::full_mesh();
    config.requires_vc2 = true;
    for (const auto& test : cases) {
        const auto shape =
            router_vc_shape(test.topology, test.facing, chip_with_z(test.z_capability), test.express, &config);
        EXPECT_EQ(shape.num_vcs, 3);
        EXPECT_EQ(shape.sender_counts[2], 1);
        EXPECT_EQ(shape.flat_sender_id(2, 0), test.expected_sender);
        EXPECT_EQ(shape.receiver_counts[2], test.has_receiver ? 1u : 0u);
        if (test.has_receiver) {
            EXPECT_EQ(shape.flat_receiver_id(2, 0), 2);
        } else {
            EXPECT_THROW((void)shape.flat_receiver_id(2, 0), std::exception);
        }
    }
}

// VC2 line mode (VC2_LINE_BRINGUP.md): a cardinal router gets VC2's forwarding channel after its worker, fed by the
// router facing the opposite way, and turns VC2 only straight on.
TEST_F(RouterWiringRulesTest, VC2LineModeShapeAndTurns) {
    auto config = IntermeshVCConfig::disabled();
    config.requires_vc2 = true;
    config.vc2_line = true;
    for (const auto facing : {RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W}) {
        const auto archetype = router_archetype(Topology::Mesh, facing, chip_with_z(std::nullopt), false, &config);
        EXPECT_EQ(archetype.shape.num_vcs, 3);
        EXPECT_EQ(archetype.shape.sender_counts[2], 2u);
        EXPECT_EQ(archetype.shape.flat_sender_id(2, 0), 4u);
        EXPECT_EQ(archetype.shape.flat_sender_id(2, 1), 5u);
        EXPECT_EQ(archetype.shape.receiver_counts[2], 1u);
        ASSERT_EQ(archetype.turns[2].size(), 1u);
        EXPECT_EQ(archetype.turns[2][0].target_vc, 2u);
        EXPECT_EQ(archetype.turns[2][0].target_direction, get_opposite_direction(facing));
    }
    // Neighbour mode: the worker only, no VC2 turns.
    config.vc2_line = false;
    const auto neighbour = router_archetype(Topology::Mesh, RoutingDirection::E, chip_with_z(std::nullopt), false, &config);
    EXPECT_EQ(neighbour.shape.sender_counts[2], 1u);
    EXPECT_TRUE(neighbour.turns[2].empty());
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


// Line VCs on 1D fabrics (TT_METAL_FABRIC_LINE_VCS, LINE_VCS_HOST_DESIGN.md): every line VC gets VC0's 1D shape (a
// worker and a forwarding channel, a receiver) and turns straight on only.
TEST_F(RouterWiringRulesTest, LineVcsOn1DShapeAndTurns) {
    for (const auto topology : {Topology::Linear, Topology::Ring}) {
        for (const int line_vcs : {2, 3}) {
            auto config = IntermeshVCConfig::disabled();
            config.vc_routing = vc_routings_of(config, topology, line_vcs);
            config.requires_vc2 = config.vc_present(2);  // as compute_intermesh_vc_config sets it
            for (const auto facing : {RoutingDirection::E, RoutingDirection::W}) {
                const auto archetype = router_archetype(topology, facing, chip_with_z(std::nullopt), false, &config);
                EXPECT_EQ(archetype.shape.num_vcs, static_cast<uint32_t>(line_vcs));
                for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
                    const bool line = vc < static_cast<uint32_t>(line_vcs);
                    EXPECT_EQ(archetype.shape.sender_counts[vc], line ? 2u : 0u) << "VC" << vc;
                    EXPECT_EQ(archetype.shape.receiver_counts[vc], line ? 1u : 0u) << "VC" << vc;
                    if (line) {
                        EXPECT_EQ(archetype.shape.flat_sender_id(vc, 0), 2 * vc);
                        ASSERT_EQ(archetype.turns[vc].size(), 1u) << "VC" << vc;
                        EXPECT_EQ(archetype.turns[vc][0].target_vc, vc);
                        EXPECT_EQ(archetype.turns[vc][0].target_direction, get_opposite_direction(facing));
                    } else {
                        EXPECT_TRUE(archetype.turns[vc].empty()) << "VC" << vc;
                    }
                }
            }
        }
    }
}

// Each VC's routing mode from the format (the topology) and the options.
TEST_F(RouterWiringRulesTest, VcRoutingsFromFormatAndOptions) {
    using builder_config::VcRouting;
    using Routings = builder_config::VcRoutings;
    auto config = IntermeshVCConfig::disabled();
    EXPECT_EQ(vc_routings_of(config, Topology::Linear, 1), (Routings{VcRouting::LINE, VcRouting::OFF, VcRouting::OFF}));
    EXPECT_EQ(vc_routings_of(config, Topology::Ring, 3), (Routings{VcRouting::LINE, VcRouting::LINE, VcRouting::LINE}));
    EXPECT_EQ(
        vc_routings_of(config, Topology::Mesh, 3), (Routings{VcRouting::FULL_2D, VcRouting::OFF, VcRouting::OFF}));
    config = IntermeshVCConfig::full_mesh();
    config.requires_vc2 = true;
    EXPECT_EQ(
        vc_routings_of(config, Topology::Mesh, 1),
        (Routings{VcRouting::FULL_2D, VcRouting::INTER_MESH, VcRouting::NEIGHBOUR}));
    config.vc2_line = true;
    EXPECT_EQ(
        vc_routings_of(config, Topology::Torus, 1),
        (Routings{VcRouting::FULL_2D, VcRouting::INTER_MESH, VcRouting::LINE}));
}

// The validation table: every combination it offers passes, every other one is refused.
TEST_F(RouterWiringRulesTest, VcRoutingValidationTable) {
    using builder_config::VcRouting;
    const auto check = [](builder_config::VcRoutings routing,
                          Topology topology,
                          bool requires_vc1 = false,
                          FabricTensixConfig tensix = FabricTensixConfig::DISABLED,
                          tt::ARCH arch = tt::ARCH::BLACKHOLE) {
        auto config = requires_vc1 ? IntermeshVCConfig::full_mesh() : IntermeshVCConfig::disabled();
        config.vc_routing = routing;
        validate_vc_routings(config, topology, tensix, arch);
    };
    constexpr auto OFF = VcRouting::OFF;
    constexpr auto LINE = VcRouting::LINE;
    constexpr auto FULL_2D = VcRouting::FULL_2D;
    // Offered.
    EXPECT_NO_THROW(check({FULL_2D, OFF, OFF}, Topology::Mesh));
    EXPECT_NO_THROW(check({FULL_2D, VcRouting::INTER_MESH, LINE}, Topology::Mesh, true));
    EXPECT_NO_THROW(check({FULL_2D, OFF, VcRouting::NEIGHBOUR}, Topology::Torus));
    EXPECT_NO_THROW(check({LINE, OFF, OFF}, Topology::NeighborExchange));
    EXPECT_NO_THROW(check({LINE, LINE, OFF}, Topology::Linear));
    EXPECT_NO_THROW(check({LINE, LINE, LINE}, Topology::Ring));
    EXPECT_NO_THROW(check({LINE, OFF, OFF}, Topology::Linear, false, FabricTensixConfig::MUX));
    // Modes the format does not offer.
    EXPECT_ANY_THROW(check({LINE, OFF, OFF}, Topology::Mesh));
    EXPECT_ANY_THROW(check({FULL_2D, LINE, OFF}, Topology::Mesh));
    EXPECT_ANY_THROW(check({FULL_2D, OFF, VcRouting::INTER_MESH}, Topology::Mesh));
    EXPECT_ANY_THROW(check({FULL_2D, OFF, OFF}, Topology::Linear));
    EXPECT_ANY_THROW(check({LINE, VcRouting::INTER_MESH, OFF}, Topology::Linear));
    EXPECT_ANY_THROW(check({LINE, OFF, VcRouting::NEIGHBOUR}, Topology::Ring));
    // What several line VCs need.
    EXPECT_ANY_THROW(check({LINE, OFF, LINE}, Topology::Linear));
    EXPECT_ANY_THROW(check({LINE, LINE, OFF}, Topology::NeighborExchange));
    EXPECT_ANY_THROW(check({LINE, LINE, OFF}, Topology::Linear, true));
    EXPECT_ANY_THROW(check({LINE, LINE, OFF}, Topology::Ring, false, FabricTensixConfig::MUX));
    EXPECT_ANY_THROW(check({LINE, LINE, OFF}, Topology::Ring, false, FabricTensixConfig::UDM));
    EXPECT_ANY_THROW(
        check({LINE, LINE, OFF}, Topology::Ring, false, FabricTensixConfig::DISABLED, tt::ARCH::WORMHOLE_B0));
}

// NoC transaction IDs per receiver channel: within the 16 IDs, a power of 2, today's counts for one to three channels.
TEST_F(RouterWiringRulesTest, TransactionIdsPerReceiverChannel) {
    using builder_config::transaction_ids_per_receiver_channel;
    EXPECT_EQ(transaction_ids_per_receiver_channel(1, false), 16u);
    EXPECT_EQ(transaction_ids_per_receiver_channel(1, true), 8u);
    EXPECT_EQ(transaction_ids_per_receiver_channel(2, true), 8u);
    EXPECT_EQ(transaction_ids_per_receiver_channel(2, false), 4u);
    EXPECT_EQ(transaction_ids_per_receiver_channel(3, true), 4u);
    EXPECT_EQ(transaction_ids_per_receiver_channel(3, false), 4u);
    for (size_t channels = 1; channels <= 3; ++channels) {
        for (const bool deadlock_avoidance : {false, true}) {
            const uint32_t ids = transaction_ids_per_receiver_channel(channels, deadlock_avoidance);
            EXPECT_LE(channels * ids, 16u);
            EXPECT_EQ(ids & (ids - 1), 0u);
        }
    }
}

}  // namespace tt::tt_fabric
