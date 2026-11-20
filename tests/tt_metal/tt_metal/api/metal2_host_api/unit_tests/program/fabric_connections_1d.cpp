// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fabric 1D connections (ProgramSpec::fabric_connections): the per-node connection args that a bound kernel
// receives. Each target of a binding owns kWordsPerTarget1D runtime-arg words on every node of the kernel:
// [connected, then the connection args that append_fabric_connection_rt_args gives].

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <set>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "metal2_host_api/test_helpers/fabric_mock_fixture.hpp"

namespace tt::tt_metal {
namespace {

using test_helpers::LegacyConnectionArgsForTest;
using test_helpers::MakeFabricConnectionForTest;
using test_helpers::MakeSpecWithFabricConnectionForTest;
using test_helpers::ProgramSpecFabric1DTest;
using test_helpers::WriterRuntimeArgsForTest;

constexpr size_t kWordsPerTarget1D = 4;

// Not a public API: the ids of every semaphore of the workload's Program on `node`.
std::vector<uint32_t> SemaphoreIdsOnNodeForTest(distributed::MeshWorkload& workload, const NodeCoord& node) {
    const auto semaphores = workload.get_programs().begin()->second.impl().semaphores_on_core(node, CoreType::WORKER);
    std::vector<uint32_t> ids;
    std::ranges::transform(
        semaphores, std::back_inserter(ids), [](const auto& semaphore) { return semaphore.get().id(); });
    return ids;
}

TEST_F(ProgramSpecFabric1DTest, CPU_ConnectionToNeighbourMatchesLegacyConnectionArgs) {
    // A connected target gets a "connected" word, then the same args as append_fabric_connection_rt_args.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(Fabric1DTargets{fixture_neighbour_direction()}));
    distributed::MeshWorkload workload(fixture_device(), spec);
    workload.get_programs().begin()->second.set_program_run_args(ProgramRunArgs{});
    std::vector<uint32_t> expected = {1u};
    std::ranges::copy(
        LegacyConnectionArgsForTest(fixture_device(), fixture_neighbour_fabric_node_id()),
        std::back_inserter(expected));
    EXPECT_EQ(WriterRuntimeArgsForTest(workload, NodeCoord{0, 0}), expected);
}

TEST_F(ProgramSpecFabric1DTest, CPU_ConnectionArgsAreSetWithoutSettingRunArgs) {
    // A Program enqueued without Program::set_program_run_args gives its kernel the connection args.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(Fabric1DTargets{fixture_neighbour_direction()}));
    distributed::MeshWorkload workload(fixture_device(), spec);
    distributed::EnqueueMeshWorkload(fixture_device().mesh_command_queue(), workload, false);
    std::vector<uint32_t> expected = {1u};
    std::ranges::copy(
        LegacyConnectionArgsForTest(fixture_device(), fixture_neighbour_fabric_node_id()),
        std::back_inserter(expected));
    EXPECT_EQ(WriterRuntimeArgsForTest(workload, NodeCoord{0, 0}), expected);
}

TEST_F(ProgramSpecFabric1DTest, CPU_TargetWithoutNeighbourIsNotConnected) {
    // A direction with no neighbouring chip is not connected on that device: its words are zero.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(MakeFabricConnectionForTest(
        Fabric1DTargets{fixture_neighbour_direction(), fixture_direction_without_neighbour()}));
    distributed::MeshWorkload workload(fixture_device(), spec);
    workload.get_programs().begin()->second.set_program_run_args(ProgramRunArgs{});
    const std::vector<uint32_t> args = WriterRuntimeArgsForTest(workload, NodeCoord{0, 0});
    ASSERT_EQ(args.size(), 2 * kWordsPerTarget1D);
    EXPECT_THAT(std::vector<uint32_t>(args.begin() + kWordsPerTarget1D, args.end()), ::testing::Each(0u));
}

TEST_F(ProgramSpecFabric1DTest, CPU_NodeWithoutLinkIsNotConnected) {
    // A node of the binding kernel that is not in links has no connection: its words are zero.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(Fabric1DTargets{fixture_neighbour_direction()}));
    spec.work_units = {test_helpers::MakeMinimalWorkUnit("work_unit", NodeRange({0, 0}, {1, 0}), {"writer"})};
    distributed::MeshWorkload workload(fixture_device(), spec);
    workload.get_programs().begin()->second.set_program_run_args(ProgramRunArgs{});
    EXPECT_THAT(
        WriterRuntimeArgsForTest(workload, NodeCoord{1, 0}),
        ::testing::AllOf(::testing::SizeIs(kWordsPerTarget1D), ::testing::Each(0u)));
}

TEST_F(ProgramSpecFabric1DTest, CPU_ConnectionSemaphoresDoNotShareIdsWithSemaphoreSpecs) {
    // The runtime's connection semaphores and the Program's own semaphores have different ids on a node.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(Fabric1DTargets{fixture_neighbour_direction()}));
    spec.semaphores = {SemaphoreSpec{.unique_id = SemaphoreSpecName{"sem"}, .target_nodes = NodeCoord{0, 0}}};
    spec.kernels[0].semaphore_bindings = {
        SemaphoreBinding{.semaphore_spec_name = SemaphoreSpecName{"sem"}, .accessor_name = "sem"}};
    distributed::MeshWorkload workload(fixture_device(), spec);
    const std::vector<uint32_t> ids = SemaphoreIdsOnNodeForTest(workload, NodeCoord{0, 0});
    EXPECT_EQ(std::set<uint32_t>(ids.begin(), ids.end()).size(), 3u);
}

TEST_F(ProgramSpecFabric1DTest, CPU_ConnectionArgsSurviveSettingRunArgsAgain) {
    // Setting the Program's run args again keeps the connection args.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(Fabric1DTargets{fixture_neighbour_direction()}));
    distributed::MeshWorkload workload(fixture_device(), spec);
    Program& program = workload.get_programs().begin()->second;
    program.set_program_run_args(ProgramRunArgs{});
    const std::vector<uint32_t> first = WriterRuntimeArgsForTest(workload, NodeCoord{0, 0});
    program.set_program_run_args(ProgramRunArgs{});
    EXPECT_EQ(WriterRuntimeArgsForTest(workload, NodeCoord{0, 0}), first);
}

}  // namespace
}  // namespace tt::tt_metal
