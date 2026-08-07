// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Regression for the edge-capability truth table in
// GALAXY_BUILDER_ROUTING_CONFIG_CONTRACT.md section 4.3.
//
// The property under test is that a direction letter selects an output while capability selects
// transport behaviour, and the two are independent: Z does not imply intermesh, and intermesh is not
// confined to Z.

#include <gtest/gtest.h>

#include <enchantum/enchantum.hpp>

#include "tt_metal/fabric/builder/fabric_edge_capability.hpp"

namespace tt::tt_fabric {
namespace {

constexpr bool k_express_on = true;
constexpr bool k_express_off = false;

FabricNodeId node(uint32_t mesh, uint32_t chip) { return FabricNodeId(MeshId{mesh}, chip); }

TEST(FabricEdgeCapabilityTest, CrossMeshIsIntermeshOnEveryDirection) {
    const auto local = node(0, 4);
    const auto remote = node(1, 9);

    // An intermesh edge is not confined to Z; it can sit on any compass letter.
    for (const auto direction :
         {RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W, RoutingDirection::Z}) {
        EXPECT_EQ(classify_fabric_edge(local, remote, direction, k_express_off), EdgeCapability::INTERMESH);
        EXPECT_EQ(classify_fabric_edge(local, remote, direction, k_express_on), EdgeCapability::INTERMESH);
    }
}

TEST(FabricEdgeCapabilityTest, SameMeshCardinalIsIntramesh) {
    const auto local = node(0, 4);
    const auto remote = node(0, 5);

    for (const auto direction : {RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W}) {
        // Express enablement must not change how an ordinary cardinal edge is classified.
        EXPECT_EQ(classify_fabric_edge(local, remote, direction, k_express_off), EdgeCapability::INTRAMESH_CARDINAL);
        EXPECT_EQ(classify_fabric_edge(local, remote, direction, k_express_on), EdgeCapability::INTRAMESH_CARDINAL);
    }
}

TEST(FabricEdgeCapabilityTest, SameMeshZWithExpressIsExpress) {
    EXPECT_EQ(
        classify_fabric_edge(node(0, 2), node(0, 5), RoutingDirection::Z, k_express_on),
        EdgeCapability::INTRAMESH_EXPRESS);
}

TEST(FabricEdgeCapabilityTest, SameMeshZWithoutExpressIsRejected) {
    // Topology intent and the neighbor graph disagree. Classifying it as either cardinal or intermesh
    // would hide that, so the configuration fails instead.
    EXPECT_ANY_THROW(classify_fabric_edge(node(0, 2), node(0, 5), RoutingDirection::Z, k_express_off));
}

TEST(FabricEdgeCapabilityTest, ExpressAndIntermeshAreDistinguishedOnTheSameChip) {
    // The case the old "any active Z channel means intermesh" test could not represent: one chip with
    // a same-mesh express chord on Z and an intermesh edge on a cardinal port.
    const auto local = node(0, 2);
    EXPECT_EQ(
        classify_fabric_edge(local, node(0, 5), RoutingDirection::Z, k_express_on), EdgeCapability::INTRAMESH_EXPRESS);
    EXPECT_EQ(classify_fabric_edge(local, node(3, 0), RoutingDirection::E, k_express_on), EdgeCapability::INTERMESH);
}

TEST(FabricEdgeCapabilityTest, CapabilityNamesAreStable) {
    // These strings appear in configuration failures, so keep them recognizable.
    EXPECT_EQ(enchantum::to_string(EdgeCapability::INTRAMESH_CARDINAL), "INTRAMESH_CARDINAL");
    EXPECT_EQ(enchantum::to_string(EdgeCapability::INTRAMESH_EXPRESS), "INTRAMESH_EXPRESS");
    EXPECT_EQ(enchantum::to_string(EdgeCapability::INTERMESH), "INTERMESH");
}

TEST(FabricEdgeCapabilityTest, CapabilitiesAtChecksThePortDomain) {
    // The five ports round-trip through at(); C and NONE are not ports and must never index the
    // set -- the array has five slots and the enum has seven values, so the check is the point.
    PerDirectionCapabilities caps;
    for (const auto direction :
         {RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W, RoutingDirection::Z}) {
        caps.at(direction) = EdgeCapability::INTRAMESH_CARDINAL;
    }
    for (const auto direction :
         {RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W, RoutingDirection::Z}) {
        EXPECT_EQ(caps.at(direction), EdgeCapability::INTRAMESH_CARDINAL)
            << "direction " << enchantum::to_string(direction);
    }
    // A direction never written is absent.
    EXPECT_FALSE(PerDirectionCapabilities().at(RoutingDirection::Z).has_value());

    EXPECT_ANY_THROW(caps.at(RoutingDirection::C));
    EXPECT_ANY_THROW(caps.at(RoutingDirection::NONE));
}

}  // namespace
}  // namespace tt::tt_fabric
