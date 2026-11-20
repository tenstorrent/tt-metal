// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fabric 2D connections (ProgramSpec::fabric_connections with peer targets): the per-node connection args that a
// bound kernel receives. Each target of a binding owns kWordsPerTarget2D runtime-arg words on every node of the kernel:
// [connected, the connection args that append_fabric_connection_rt_args gives, peer chip id, peer mesh id].

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <vector>

#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "metal2_host_api/test_helpers/fabric_mock_fixture.hpp"

namespace tt::tt_metal {
namespace {

using test_helpers::LegacyConnectionArgsForTest;
using test_helpers::MakeFabricConnectionForTest;
using test_helpers::MakeSpecWithFabricConnectionForTest;
using test_helpers::ProgramSpecFabric2DTest;
using test_helpers::WriterRuntimeArgsForTest;

constexpr size_t kWordsPerTarget2D = 6;

TEST_F(ProgramSpecFabric2DTest, CPU_ConnectionToPeerMatchesLegacyConnectionArgs) {
    // A peer target gets a "connected" word, the args of append_fabric_connection_rt_args, then the peer's ids.
    const tt_fabric::FabricNodeId peer = fixture_neighbour_fabric_node_id();
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(MakeFabricConnectionForTest(Fabric2DTargets{peer}));
    distributed::MeshWorkload workload(fixture_device(), spec);
    workload.get_programs().begin()->second.set_program_run_args(ProgramRunArgs{});
    std::vector<uint32_t> expected = {1u};
    std::ranges::copy(LegacyConnectionArgsForTest(fixture_device(), peer), std::back_inserter(expected));
    expected.insert(expected.end(), {peer.chip_id, *peer.mesh_id});
    EXPECT_EQ(WriterRuntimeArgsForTest(workload, NodeCoord{0, 0}), expected);
}

TEST_F(ProgramSpecFabric2DTest, CPU_NodeWithoutLinkIsNotConnectedIn2D) {
    // A node of the binding kernel that is not in links has no connection: its words are zero.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(Fabric2DTargets{fixture_neighbour_fabric_node_id()}));
    spec.work_units = {test_helpers::MakeMinimalWorkUnit("work_unit", NodeRange({0, 0}, {1, 0}), {"writer"})};
    distributed::MeshWorkload workload(fixture_device(), spec);
    workload.get_programs().begin()->second.set_program_run_args(ProgramRunArgs{});
    EXPECT_THAT(
        WriterRuntimeArgsForTest(workload, NodeCoord{1, 0}),
        ::testing::AllOf(::testing::SizeIs(kWordsPerTarget2D), ::testing::Each(0u)));
}

TEST_F(ProgramSpecFabric2DTest, CPU_ConnectionToOwnDeviceFails) {
    // A peer target is another fabric node.
    ProgramSpec spec =
        MakeSpecWithFabricConnectionForTest(MakeFabricConnectionForTest(Fabric2DTargets{fixture_fabric_node_id()}));
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_device(), spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("is the Program's own fabric node")));
}

TEST_F(ProgramSpecFabric2DTest, CPU_ConnectionToPeerOutsideTheSystemFails) {
    // A peer target is a fabric node of the system.
    const tt_fabric::FabricNodeId outside_peer(fixture_fabric_node_id().mesh_id, 1000);
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(MakeFabricConnectionForTest(Fabric2DTargets{outside_peer}));
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_device(), spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("is not a fabric node of the system")));
}

}  // namespace
}  // namespace tt::tt_metal
