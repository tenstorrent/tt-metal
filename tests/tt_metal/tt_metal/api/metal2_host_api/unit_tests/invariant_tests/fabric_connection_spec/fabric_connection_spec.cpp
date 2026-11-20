// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Local invariants of FabricConnectionSpec: its targets, and that their kind matches the fabric mode.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <stdexcept>

#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "metal2_host_api/test_helpers/fabric_mock_fixture.hpp"

namespace tt::tt_metal {
namespace {

using test_helpers::MakeFabricConnectionForTest;
using test_helpers::MakeSpecWithFabricConnectionForTest;
using test_helpers::ProgramSpecFabric1DTest;
using test_helpers::ProgramSpecFabric2DTest;
using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

TEST_F(ProgramSpecFabric1DTest, CPU_FabricConnectionToNeighbourDirectionIsAccepted) {
    // A 1D connection toward the neighbouring chip is a valid spec.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(Fabric1DTargets{fixture_neighbour_direction()}));
    EXPECT_NO_THROW(distributed::MeshWorkload(fixture_device(), spec));
}

TEST_F(ProgramSpecFabric1DTest, CPU_FabricConnectionWithoutTargetsFails) {
    // A fabric connection spec has at least one target.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(MakeFabricConnectionForTest(Fabric1DTargets{}));
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_device(), spec); },
        ThrowsMessage<std::runtime_error>(HasSubstr("FabricConnectionSpec 'conn' has no targets")));
}

TEST_F(ProgramSpecFabric1DTest, CPU_FabricConnectionWithoutLinksFails) {
    // A fabric connection spec gives at least one node a link.
    FabricConnectionSpec connection = MakeFabricConnectionForTest(Fabric1DTargets{fixture_neighbour_direction()});
    connection.links = {};
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(connection);
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_device(), spec); },
        ThrowsMessage<std::runtime_error>(HasSubstr("FabricConnectionSpec 'conn' has no links")));
}

TEST_F(ProgramSpecFabric1DTest, CPU_FabricConnectionWithRepeatedDirectionFails) {
    // A fabric connection spec names each target once.
    const tt_fabric::RoutingDirection direction = fixture_neighbour_direction();
    ProgramSpec spec =
        MakeSpecWithFabricConnectionForTest(MakeFabricConnectionForTest(Fabric1DTargets{direction, direction}));
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_device(), spec); },
        ThrowsMessage<std::runtime_error>(HasSubstr("FabricConnectionSpec 'conn' has a target more than once")));
}

TEST_F(ProgramSpecFabric1DTest, CPU_FabricConnectionToCentreDirectionFails) {
    // A direction target is N, E, S, W or Z.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(Fabric1DTargets{tt_fabric::RoutingDirection::C}));
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_device(), spec); },
        ThrowsMessage<std::runtime_error>(HasSubstr("a direction target is N, E, S, W or Z")));
}

TEST_F(ProgramSpecFabric1DTest, CPU_FabricConnectionWith2DTargetsOn1DFabricFails) {
    // Peer targets need a 2D fabric.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(Fabric2DTargets{fixture_neighbour_fabric_node_id()}));
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_device(), spec); },
        ThrowsMessage<std::runtime_error>(HasSubstr("has 2D targets, but the fabric is not in a 2D mode")));
}

TEST_F(ProgramSpecFabric2DTest, CPU_FabricConnectionWith1DTargetsOn2DFabricFails) {
    // Direction targets need a 1D fabric.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(Fabric1DTargets{fixture_neighbour_direction()}));
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_device(), spec); },
        ThrowsMessage<std::runtime_error>(HasSubstr("has 1D targets, but the fabric is not in a 1D mode")));
}

}  // namespace
}  // namespace tt::tt_metal
