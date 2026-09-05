// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Exhaustive properties for express-routing connection wiring.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <enchantum/enchantum.hpp>
#include <set>

#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "tt_metal/fabric/builder/router_wiring_rules.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"

namespace tt::tt_fabric {
namespace {

constexpr bool k_express = true;
constexpr bool k_no_express = false;

const IntermeshVCConfig k_full_mesh = IntermeshVCConfig::full_mesh();

// The full input domains of the wiring primitive, swept by the property tests below.
constexpr std::array<RoutingDirection, 5> k_all_directions = {
    RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W, RoutingDirection::Z};
constexpr std::array<EdgeCapability, 3> k_all_capabilities = {
    EdgeCapability::INTRAMESH_CARDINAL, EdgeCapability::INTRAMESH_EXPRESS, EdgeCapability::INTERMESH};
constexpr std::array<ZPortRole, 3> k_all_z_roles = {
    ZPortRole::NONE, ZPortRole::INTERMESH_BOUNDARY, ZPortRole::EXPRESS_CHORD};

// Cardinals are ordinary same-mesh edges; the Z capability follows the chip role.
std::optional<EdgeCapability> egress_capability_on(RoutingDirection egress, ZPortRole chip_z_role) {
    if (egress != RoutingDirection::Z) {
        return EdgeCapability::INTRAMESH_CARDINAL;
    }
    switch (chip_z_role) {
        case ZPortRole::INTERMESH_BOUNDARY: return EdgeCapability::INTERMESH;
        case ZPortRole::EXPRESS_CHORD: return EdgeCapability::INTRAMESH_EXPRESS;
        case ZPortRole::NONE: break;
    }
    return std::nullopt;
}

// wires_into with the egress side resolved against that chip.
bool wires_into_chip(
    RoutingDirection producer_direction,
    EdgeCapability producer_capability,
    RoutingDirection egress_direction,
    ZPortRole chip_z_role,
    bool express_routing_enabled,
    uint32_t vc) {
    return wires_into(
        producer_direction,
        producer_capability,
        egress_direction,
        egress_capability_on(egress_direction, chip_z_role),
        chip_z_role,
        express_routing_enabled,
        vc);
}

// One chip, from the role its Z port plays and an optional cardinal seam.
PerDirectionCapabilities chip_with(ZPortRole chip_z_role, std::optional<RoutingDirection> seam_facing = std::nullopt) {
    PerDirectionCapabilities caps;
    for (const auto direction : {RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W}) {
        caps.at(direction) = EdgeCapability::INTRAMESH_CARDINAL;
    }
    caps.at(RoutingDirection::Z) = egress_capability_on(RoutingDirection::Z, chip_z_role);
    if (seam_facing.has_value()) {
        caps.at(*seam_facing) = EdgeCapability::INTERMESH;
    }
    return caps;
}

// The chip a router of this facing and capability sits on: its own edge as given, the rest
// ordinary, and the Z port as its role says.
PerDirectionCapabilities chip_facing(RoutingDirection facing, EdgeCapability capability, ZPortRole chip_z_role) {
    auto caps = chip_with(chip_z_role);
    caps.at(facing) = capability;
    return caps;
}

std::set<RoutingDirection> target_directions(const RouterTurnSet& turn_set, uint32_t vc) {
    std::set<RoutingDirection> dirs;
    for (const auto& target : turn_set[vc]) {
        if (target.target_direction.has_value()) {
            dirs.insert(*target.target_direction);
        }
    }
    return dirs;
}

TEST(ExpressConnectionWiringTest, NoRouterIsWiredBackOverItsOwnLink) {
    // U-turn rejection precedes capability classification.
    for (const auto ingress : k_all_directions) {
        for (const auto capability : k_all_capabilities) {
            for (const auto role : k_all_z_roles) {
                for (const bool express : {false, true}) {
                    for (const uint32_t vc : {0u, 1u}) {
                        EXPECT_FALSE(wires_into_chip(ingress, capability, ingress, role, express, vc))
                            << "ingress " << enchantum::to_string(ingress) << " (" << enchantum::to_string(capability)
                            << "), chip role " << enchantum::to_string(role) << ", express " << express << ", vc "
                            << vc;
                    }
                }
            }
        }
    }
}

TEST(ExpressConnectionWiringTest, BoundaryProducerFeedsNothingOnVC0InEitherMode) {
    // A boundary receiver crosses VC0 traffic over and fans out only from VC1.
    for (const auto express : {false, true}) {
        for (const auto egress : {RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W}) {
            EXPECT_FALSE(wires_into_chip(
                RoutingDirection::Z,
                EdgeCapability::INTERMESH,
                egress,
                ZPortRole::INTERMESH_BOUNDARY,
                express,
                /*vc=*/0))
                << "express=" << express << " egress " << enchantum::to_string(egress);
            EXPECT_TRUE(wires_into_chip(
                RoutingDirection::Z,
                EdgeCapability::INTERMESH,
                egress,
                ZPortRole::INTERMESH_BOUNDARY,
                express,
                /*vc=*/1))
                << "express=" << express << " egress " << enchantum::to_string(egress);
        }
    }
}

TEST(ExpressConnectionWiringTest, IntrameshXWiresIntoAnIntermeshEgressOnAnyLetter) {
    // Leaving the mesh is exempt from the intramesh X-to-Y restriction, regardless of seam letter.
    for (const auto x : {RoutingDirection::E, RoutingDirection::W}) {
        for (const auto seam : {RoutingDirection::N, RoutingDirection::S, RoutingDirection::Z}) {
            EXPECT_TRUE(wires_into(
                x,
                EdgeCapability::INTRAMESH_CARDINAL,
                seam,
                EdgeCapability::INTERMESH,
                seam == RoutingDirection::Z ? ZPortRole::INTERMESH_BOUNDARY : ZPortRole::EXPRESS_CHORD,
                /*express_routing_enabled=*/true,
                /*vc=*/0))
                << "X producer " << enchantum::to_string(x) << " must reach the seam on " << enchantum::to_string(seam);

            // Same letter, ordinary same-mesh edge: still unwired. The exemption is the seam's
            // capability, not a hole in dimension order.
            const auto intramesh =
                seam == RoutingDirection::Z ? EdgeCapability::INTRAMESH_EXPRESS : EdgeCapability::INTRAMESH_CARDINAL;
            EXPECT_FALSE(wires_into(
                x,
                EdgeCapability::INTRAMESH_CARDINAL,
                seam,
                intramesh,
                ZPortRole::EXPRESS_CHORD,
                /*express_routing_enabled=*/true,
                /*vc=*/0));
        }
    }
}

// --- Sender counts are the family max over facing of wired-producer arity, not constants ---

TEST(ExpressConnectionWiringTest, ExpressSenderCountsAreFamilyMaxOverFacing) {
    // E/W routers determine the five-sender family maximum.
    const auto canonical = canonical_express_endpoint_capabilities();
    EXPECT_EQ(express_vc0_producer_arity(RoutingDirection::E, canonical), 5u);
    EXPECT_EQ(express_vc0_producer_arity(RoutingDirection::W, canonical), 5u);
    EXPECT_EQ(express_vc0_producer_arity(RoutingDirection::N, canonical), 3u);
    EXPECT_EQ(express_vc0_producer_arity(RoutingDirection::S, canonical), 3u);
    EXPECT_EQ(express_vc0_producer_arity(RoutingDirection::Z, canonical), 3u);

    EXPECT_EQ(express_vc0_sender_count(), 5u);
    EXPECT_EQ(express_vc1_sender_count(), 4u);
}

TEST(ExpressConnectionWiringTest, ArityRespectsPerChipCapabilities) {
    // Per-chip arity can be narrower than, or add a landing to, the family maximum.
    auto landing = canonical_express_endpoint_capabilities();
    landing.at(RoutingDirection::E) = EdgeCapability::INTERMESH;

    EXPECT_EQ(express_vc0_producer_arity(RoutingDirection::N, landing), 4u);
    EXPECT_EQ(express_vc0_producer_arity(RoutingDirection::Z, landing), 4u);
    EXPECT_EQ(express_vc0_producer_arity(RoutingDirection::W, landing), 5u);

    auto leaf = canonical_express_endpoint_capabilities();
    leaf.at(RoutingDirection::Z) = std::nullopt;
    EXPECT_EQ(express_vc0_producer_arity(RoutingDirection::N, leaf), 2u);
    EXPECT_EQ(express_vc0_producer_arity(RoutingDirection::E, leaf), 4u);
}

// Legal input combinations for the property sweeps.
bool archetype_buildable(RoutingDirection facing, EdgeCapability capability, ZPortRole role, bool express) {
    if (facing == RoutingDirection::Z) {
        if (is_z_boundary_router(facing, capability)) {
            return role == ZPortRole::INTERMESH_BOUNDARY;
        }
        return capability == EdgeCapability::INTRAMESH_EXPRESS && role == ZPortRole::EXPRESS_CHORD && express;
    }
    return capability != EdgeCapability::INTRAMESH_EXPRESS;
}

bool turn_set_has_direction(const RouterTurnSet& turn_set, uint32_t vc, RoutingDirection direction) {
    return std::any_of(turn_set[vc].begin(), turn_set[vc].end(), [&](const ConnectionTarget& target) {
        return target.target_direction == direction;
    });
}

TEST(ExpressConnectionWiringTest, TurnSetMembershipMatchesThePrimitive) {
    // Turn-set emission and the primitive must describe the same relation.
    for (const auto facing : k_all_directions) {
        for (const auto capability : k_all_capabilities) {
            if (is_z_boundary_router(facing, capability)) {
                continue;  // the boundary template, not turn-matrix-derived
            }
            for (const auto role : k_all_z_roles) {
                for (const bool express : {false, true}) {
                    if (!archetype_buildable(facing, capability, role, express)) {
                        continue;
                    }
                    const auto turns = turn_set_for_router(
                        Topology::Torus, facing, chip_facing(facing, capability, role), express, &k_full_mesh);
                    for (const auto egress : k_all_directions) {
                        EXPECT_EQ(
                            turn_set_has_direction(turns, 0, egress),
                            wires_into_chip(facing, capability, egress, role, express, /*vc=*/0))
                            << "facing " << enchantum::to_string(facing) << " (" << enchantum::to_string(capability)
                            << "), chip role " << enchantum::to_string(role) << ", express " << express << ", egress "
                            << enchantum::to_string(egress);
                    }
                }
            }
        }
    }
}

TEST(ExpressConnectionWiringTest, OnlyTheBoundaryProducerIsVcSensitive) {
    // Only the boundary producer changes wiring by carrier VC.
    for (const auto producer : k_all_directions) {
        for (const auto capability : k_all_capabilities) {
            if (is_z_boundary_router(producer, capability)) {
                continue;  // the one intended exception
            }
            for (const auto egress : k_all_directions) {
                for (const auto role : k_all_z_roles) {
                    for (const bool express : {false, true}) {
                        EXPECT_EQ(
                            wires_into_chip(producer, capability, egress, role, express, /*vc=*/0),
                            wires_into_chip(producer, capability, egress, role, express, /*vc=*/1))
                            << "producer " << enchantum::to_string(producer) << " (" << enchantum::to_string(capability)
                            << ") -> egress " << enchantum::to_string(egress) << ", chip role "
                            << enchantum::to_string(role) << ", express " << express;
                    }
                }
            }
        }
    }
}

TEST(ExpressConnectionWiringTest, Vc1CarriesEveryVc0OutputExceptTheBoundaryTarget) {
    // VC1 mirrors VC0 outputs except for its worker and optional boundary pass-through.
    const auto pass_through = IntermeshVCConfig::full_mesh_with_pass_through();
    const struct {
        const IntermeshVCConfig* config;
        bool pass_through;
    } vc_cases[] = {{nullptr, false}, {&k_full_mesh, false}, {&pass_through, true}};
    for (const auto facing : k_all_directions) {
        for (const auto capability : k_all_capabilities) {
            if (is_z_boundary_router(facing, capability)) {
                continue;  // the boundary template has its own VC1 shape
            }
            for (const auto role : k_all_z_roles) {
                for (const bool express : {false, true}) {
                    if (!archetype_buildable(facing, capability, role, express)) {
                        continue;
                    }
                    for (const auto& vc_case : vc_cases) {
                        const auto turns = turn_set_for_router(
                            Topology::Torus, facing, chip_facing(facing, capability, role), express, vc_case.config);
                        for (uint32_t vc : {0u, 1u}) {
                            for (const auto& target : turns[vc]) {
                                EXPECT_EQ(target.target_vc, vc);
                            }
                        }
                        std::set<RoutingDirection> expected = target_directions(turns, 0);
                        if (vc_case.config == nullptr) {
                            expected.clear();
                        } else if (role == ZPortRole::INTERMESH_BOUNDARY && !vc_case.pass_through) {
                            expected.erase(RoutingDirection::Z);
                        }
                        EXPECT_EQ(target_directions(turns, 1), expected)
                            << "facing " << enchantum::to_string(facing) << " (" << enchantum::to_string(capability)
                            << "), chip role " << enchantum::to_string(role) << ", express " << express;
                    }
                }
            }
        }
    }
}

TEST(ExpressConnectionWiringTest, SameMeshXIngressNeverReentersY) {
    // Dimension order forbids a same-mesh X producer from re-entering intramesh Y.
    for (const auto producer : {RoutingDirection::E, RoutingDirection::W}) {
        for (const auto capability : {EdgeCapability::INTRAMESH_CARDINAL, EdgeCapability::INTRAMESH_EXPRESS}) {
            for (const auto egress : {RoutingDirection::N, RoutingDirection::S, RoutingDirection::Z}) {
                for (const auto role : k_all_z_roles) {
                    if (egress == RoutingDirection::Z && role == ZPortRole::INTERMESH_BOUNDARY) {
                        continue;
                    }
                    for (const uint32_t vc : {0u, 1u}) {
                        EXPECT_FALSE(wires_into_chip(producer, capability, egress, role, k_express, vc))
                            << "producer " << enchantum::to_string(producer) << " (" << enchantum::to_string(capability)
                            << ") -> egress " << enchantum::to_string(egress) << ", chip role "
                            << enchantum::to_string(role) << ", vc " << vc;
                    }
                }
            }
        }
    }
}

TEST(ExpressConnectionWiringTest, LandingXIngressIsExemptFromDimensionOrder) {
    // A boundary landing is a route root, not a packet mid-X-phase: an INTERMESH producer on an X
    // port keeps every non-self egress the chip has, intramesh Y included.
    for (const auto producer : {RoutingDirection::E, RoutingDirection::W}) {
        for (const auto role : k_all_z_roles) {
            for (const uint32_t vc : {0u, 1u}) {
                for (const auto egress :
                     {RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W}) {
                    if (egress == producer) {
                        continue;
                    }
                    EXPECT_TRUE(wires_into_chip(producer, EdgeCapability::INTERMESH, egress, role, k_express, vc))
                        << "landing producer " << enchantum::to_string(producer) << " -> egress "
                        << enchantum::to_string(egress) << ", chip role " << enchantum::to_string(role) << ", vc "
                        << vc;
                }
                EXPECT_EQ(
                    wires_into_chip(producer, EdgeCapability::INTERMESH, RoutingDirection::Z, role, k_express, vc),
                    role != ZPortRole::NONE)
                    << "landing producer " << enchantum::to_string(producer) << " -> Z, chip role "
                    << enchantum::to_string(role) << ", vc " << vc;
            }
        }
    }
}

TEST(ExpressConnectionWiringTest, ZEgressIsWiredOnlyWhenTheChipHasThePort) {
    // A chip with no Z port resolves a Z target to nothing -- or worse, to an intermesh Z
    // router -- so no producer is ever wired into a Z egress on one, in either mode.
    for (const auto producer : k_all_directions) {
        for (const auto capability : k_all_capabilities) {
            for (const bool express : {false, true}) {
                for (const uint32_t vc : {0u, 1u}) {
                    EXPECT_FALSE(
                        wires_into_chip(producer, capability, RoutingDirection::Z, ZPortRole::NONE, express, vc))
                        << "producer " << enchantum::to_string(producer) << " (" << enchantum::to_string(capability)
                        << "), express " << express << ", vc " << vc;
                }
            }
        }
    }
}

TEST(ExpressConnectionWiringTest, NonExpressAdmitsEveryNonSelfCardinal) {
    // Preserve the existing producer-blind cardinal wiring in non-express mode.
    for (const auto producer : k_all_directions) {
        for (const auto capability : k_all_capabilities) {
            if (is_z_boundary_router(producer, capability)) {
                continue;
            }
            for (const auto role : k_all_z_roles) {
                for (const uint32_t vc : {0u, 1u}) {
                    for (const auto egress :
                         {RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W}) {
                        if (egress == producer) {
                            continue;
                        }
                        EXPECT_TRUE(wires_into_chip(producer, capability, egress, role, k_no_express, vc))
                            << "producer " << enchantum::to_string(producer) << " (" << enchantum::to_string(capability)
                            << ") -> egress " << enchantum::to_string(egress) << ", chip role "
                            << enchantum::to_string(role) << ", vc " << vc;
                    }
                    if (producer != RoutingDirection::Z) {  // a Z producer never faces a Z egress
                        EXPECT_EQ(
                            wires_into_chip(producer, capability, RoutingDirection::Z, role, k_no_express, vc),
                            role == ZPortRole::INTERMESH_BOUNDARY)
                            << "producer " << enchantum::to_string(producer) << " (" << enchantum::to_string(capability)
                            << ") -> Z, chip role " << enchantum::to_string(role) << ", vc " << vc;
                    }
                }
            }
        }
    }
}

}  // namespace
}  // namespace tt::tt_fabric
