// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fabric 1D connections (ProgramAdvancedOptions::fabric_connections): the per-node connection args that a bound kernel
// receives. On every node of the kernel, each target of a binding owns the runtime-arg words that
// append_fabric_connection_rt_args gives.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <tt-metalium/execution/distributed.hpp>
#include <tt-metalium/workload/mesh_workload.hpp>
#include <tt-metalium/workload/program.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "metal2_host_api/test_helpers/fabric_mock_fixture.hpp"

namespace tt::tt_metal {
namespace {

using test_helpers::KernelRuntimeArgsOnDeviceForTest;
using test_helpers::LegacyConnectionArgsForTest;
using test_helpers::MakeFabricConnectionForTest;
using test_helpers::MakeSpecWithFabricConnectionForTest;
using test_helpers::ProgramSpecFabric1DMeshTest;
using test_helpers::ProgramSpecFabric1DTest;
using test_helpers::WriterRuntimeArgsForTest;
using test_helpers::WriterRuntimeArgsOnDeviceForTest;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

// Not a public API: the ids of every semaphore of the workload's Program on `node`.
std::vector<uint32_t> SemaphoreIdsOnNodeForTest(distributed::MeshWorkload& workload, const NodeCoord& node) {
    const auto semaphores = workload.get_programs().begin()->second.impl().semaphores_on_core(node, CoreType::WORKER);
    std::vector<uint32_t> ids;
    std::ranges::transform(
        semaphores, std::back_inserter(ids), [](const auto& semaphore) { return semaphore.get().id(); });
    return ids;
}

// Not a public API: the per-device run args that the workload holds for the Program on `range`, or nullptr.
const distributed::PerDeviceRuntimeArgs* PerDeviceRunArgsForTest(
    distributed::MeshWorkload& workload, const distributed::MeshCoordinateRange& range) {
    return workload.impl().per_device_runtime_args(range);
}

TEST_F(ProgramSpecFabric1DTest, CPU_ConnectionToNeighbourMatchesLegacyConnectionArgs) {
    // A target gets the same args as append_fabric_connection_rt_args.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(experimental::Fabric1DTargets{fixture_neighbour_direction()}));
    distributed::MeshWorkload workload(fixture_device(), spec);
    workload.get_programs().begin()->second.set_program_run_args(ProgramRunArgs{});
    EXPECT_EQ(
        WriterRuntimeArgsForTest(workload, NodeCoord{0, 0}),
        LegacyConnectionArgsForTest(fixture_device(), fixture_neighbour_fabric_node_id()));
}

TEST_F(ProgramSpecFabric1DTest, CPU_ConnectionArgsAreSetWithoutSettingRunArgs) {
    // A Program enqueued without Program::set_program_run_args gives its kernel the connection args.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(experimental::Fabric1DTargets{fixture_neighbour_direction()}));
    distributed::MeshWorkload workload(fixture_device(), spec);
    distributed::EnqueueMeshWorkload(fixture_device().mesh_command_queue(), workload, false);
    EXPECT_EQ(
        WriterRuntimeArgsForTest(workload, NodeCoord{0, 0}),
        LegacyConnectionArgsForTest(fixture_device(), fixture_neighbour_fabric_node_id()));
}

TEST_F(ProgramSpecFabric1DTest, CPU_ConnectionSemaphoresDoNotShareIdsWithSemaphoreSpecs) {
    // The runtime's connection semaphores and the Program's own semaphores have different ids on a node.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(experimental::Fabric1DTargets{fixture_neighbour_direction()}));
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
        MakeFabricConnectionForTest(experimental::Fabric1DTargets{fixture_neighbour_direction()}));
    distributed::MeshWorkload workload(fixture_device(), spec);
    Program& program = workload.get_programs().begin()->second;
    program.set_program_run_args(ProgramRunArgs{});
    const std::vector<uint32_t> first = WriterRuntimeArgsForTest(workload, NodeCoord{0, 0});
    program.set_program_run_args(ProgramRunArgs{});
    EXPECT_EQ(WriterRuntimeArgsForTest(workload, NodeCoord{0, 0}), first);
}

TEST_F(ProgramSpecFabric1DTest, CPU_ConnectionOnLinkOneMatchesLegacyConnectionArgs) {
    // The link of a node selects the router, as the link of append_fabric_connection_rt_args does.
    experimental::FabricConnectionsSpec connection =
        MakeFabricConnectionForTest(experimental::Fabric1DTargets{fixture_neighbour_direction()});
    connection.links = {{NodeCoord{0, 0}, 1}};
    distributed::MeshWorkload workload(fixture_device(), MakeSpecWithFabricConnectionForTest(connection));
    workload.get_programs().begin()->second.set_program_run_args(ProgramRunArgs{});
    EXPECT_EQ(
        WriterRuntimeArgsForTest(workload, NodeCoord{0, 0}),
        LegacyConnectionArgsForTest(
            fixture_device(), fixture_neighbour_fabric_node_id(), distributed::MeshCoordinate(0, 0), /*link=*/1));
}

TEST_F(ProgramSpecFabric1DTest, CPU_ConnectionArgsFollowNamedPerNodeRuntimeArgs) {
    // A kernel gets its named per-node runtime args first, then its connection args.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(experimental::Fabric1DTargets{fixture_neighbour_direction()}));
    spec.kernels[0].runtime_arg_schema.runtime_arg_names = {"num_packets"};
    distributed::MeshWorkload workload(fixture_device(), spec);
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        {.kernel = KernelSpecName{"writer"},
         .runtime_arg_values = MakeRuntimeArgsForSingleNode(NodeCoord{0, 0}, {{"num_packets", 7u}})}};
    workload.get_programs().begin()->second.set_program_run_args(run_args);
    std::vector<uint32_t> expected = {7u};
    std::ranges::copy(
        LegacyConnectionArgsForTest(fixture_device(), fixture_neighbour_fabric_node_id()),
        std::back_inserter(expected));
    EXPECT_EQ(WriterRuntimeArgsForTest(workload, NodeCoord{0, 0}), expected);
}

TEST_F(ProgramSpecFabric1DTest, CPU_ConnectionArgsFollowPerNodeTensorBinding) {
    // A kernel gets the address of its per-node tensor binding first, then its connection args.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(experimental::Fabric1DTargets{fixture_neighbour_direction()}));
    spec.tensor_parameters = {test_helpers::MakePerNodeTensorParameter("per_node_tensor", /*num_cores=*/1)};
    test_helpers::BindTensorParameterToKernel(spec.kernels[0], "per_node_tensor", "per_node_tensor");
    distributed::MeshWorkload workload(fixture_device(), spec);
    const std::vector<uint32_t> legacy =
        LegacyConnectionArgsForTest(fixture_device(), fixture_neighbour_fabric_node_id());
    EXPECT_THAT(WriterRuntimeArgsForTest(workload, NodeCoord{0, 0}), ElementsAre(_, legacy[0], legacy[1], legacy[2]));
}

TEST_F(ProgramSpecFabric1DTest, CPU_SecondBindingFollowsEveryTargetOfTheFirstBinding) {
    // A kernel that binds two specs gets the args of each target of the first spec, then the args of the second spec.
    experimental::FabricConnectionsSpec second =
        MakeFabricConnectionForTest(experimental::Fabric1DTargets{fixture_neighbour_direction()});
    second.unique_id = experimental::FabricConnectionsSpecName{"second"};
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(MakeFabricConnectionForTest(
        experimental::Fabric1DTargets{fixture_neighbour_direction(), fixture_other_neighbour_direction()}));
    spec.advanced_options.fabric_connections.push_back(second);
    spec.kernels[0].advanced_options.fabric_connection_bindings.push_back(experimental::FabricConnectionBinding{
        .fabric_connections_spec_name = second.unique_id, .accessor_name = "second"});
    spec.advanced_options.fabric_connection_options.disable_exclusive_fabric_link_check = true;
    distributed::MeshWorkload workload(fixture_device(), spec);
    const uint32_t router = LegacyConnectionArgsForTest(fixture_device(), fixture_neighbour_fabric_node_id())[0];
    EXPECT_THAT(WriterRuntimeArgsForTest(workload, NodeCoord{0, 0}), ElementsAre(router, _, _, _, _, _, router, _, _));
}

TEST_F(ProgramSpecFabric1DMeshTest, CPU_RangedProgramConnectsEachDeviceThroughItsOwnRouter) {
    // On a Program on two devices, each device connects the direction target through its own router.
    const distributed::MeshCoordinate first(0, 0);
    const distributed::MeshCoordinate second(0, 1);
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(MakeFabricConnectionForTest(
        experimental::Fabric1DTargets{fixture_direction(first, distributed::MeshCoordinate(1, 0))}));
    const std::unordered_map<distributed::MeshCoordinateRange, ProgramSpec> specs = {
        {distributed::MeshCoordinateRange(first, second), spec}};
    distributed::MeshWorkload workload(fixture_mesh_device(), specs);
    const uint32_t first_router = LegacyConnectionArgsForTest(
        fixture_mesh_device(), fixture_fabric_node_id(distributed::MeshCoordinate(1, 0)), first)[0];
    const uint32_t second_router = LegacyConnectionArgsForTest(
        fixture_mesh_device(), fixture_fabric_node_id(distributed::MeshCoordinate(1, 1)), second)[0];
    EXPECT_THAT(WriterRuntimeArgsOnDeviceForTest(workload, first, NodeCoord{0, 0}), ElementsAre(first_router, _, _));
    EXPECT_THAT(WriterRuntimeArgsOnDeviceForTest(workload, second, NodeCoord{0, 0}), ElementsAre(second_router, _, _));
}

TEST_F(ProgramSpecFabric1DMeshTest, CPU_ProgramConstructorConnectsEachDeviceThroughItsOwnRouter) {
    // A Program built from a spec on a MeshDevice gives each device its own connection args, also to a chip outside it.
    std::shared_ptr<distributed::MeshDevice> top_row =
        fixture_mesh_device().create_submesh(distributed::MeshShape{1, 2});
    const distributed::MeshCoordinate second(0, 1);
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(MakeFabricConnectionForTest(
        experimental::Fabric1DTargets{fixture_direction(second, distributed::MeshCoordinate(1, 1))}));
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(top_row->shape()), Program(*top_row, spec));
    distributed::EnqueueMeshWorkload(top_row->mesh_command_queue(), workload, false);
    const uint32_t second_router =
        LegacyConnectionArgsForTest(*top_row, fixture_fabric_node_id(distributed::MeshCoordinate(1, 1)), second)[0];
    EXPECT_THAT(WriterRuntimeArgsOnDeviceForTest(workload, second, NodeCoord{0, 0}), ElementsAre(second_router, _, _));
}

TEST_F(ProgramSpecFabric1DMeshTest, CPU_ProgramConstructorWithDeviceWithoutNeighbourInTargetDirectionFails) {
    // Program(mesh_device, spec) resolves each direction target on every device of the MeshDevice.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(MakeFabricConnectionForTest(experimental::Fabric1DTargets{
        fixture_direction(distributed::MeshCoordinate(0, 0), distributed::MeshCoordinate(1, 0))}));
    EXPECT_THAT(
        [&] { Program(fixture_mesh_device(), spec); },
        ThrowsMessage<std::runtime_error>(HasSubstr("has no fabric neighbour in direction")));
}

TEST_F(ProgramSpecFabric1DMeshTest, CPU_ProgramOnOneDeviceGetsThatDevicesConnectionArgs) {
    // A Program on one device of the MeshDevice resolves its targets on that device only, and gets its connection args.
    const distributed::MeshCoordinate first(0, 0);
    const distributed::MeshCoordinate second(0, 1);
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(experimental::Fabric1DTargets{fixture_direction(second, first)}));
    const std::unordered_map<distributed::MeshCoordinateRange, ProgramSpec> specs = {
        {distributed::MeshCoordinateRange(second), spec}};
    distributed::MeshWorkload workload(fixture_mesh_device(), specs);
    const uint32_t second_router =
        LegacyConnectionArgsForTest(fixture_mesh_device(), fixture_fabric_node_id(first), second)[0];
    EXPECT_THAT(WriterRuntimeArgsOnDeviceForTest(workload, second, NodeCoord{0, 0}), ElementsAre(second_router, _, _));
}

TEST_F(ProgramSpecFabric1DMeshTest, CPU_SettingRunArgsKeepsEachDevicesConnectionArgs) {
    // Program::set_program_run_args drops the per-device values of run args, but not the connection args of a device.
    const distributed::MeshCoordinate second(0, 1);
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(MakeFabricConnectionForTest(
        experimental::Fabric1DTargets{fixture_direction(second, distributed::MeshCoordinate(1, 1))}));
    spec.kernels[0].runtime_arg_schema.common_runtime_arg_names = {"x"};
    const distributed::MeshCoordinateRange top_row(distributed::MeshCoordinate(0, 0), second);
    distributed::MeshWorkload workload(
        fixture_mesh_device(), std::unordered_map<distributed::MeshCoordinateRange, ProgramSpec>{{top_row, spec}});
    Program& program = workload.get_programs().begin()->second;
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {{.kernel = KernelSpecName{"writer"}, .common_runtime_arg_values = {{"x", 1u}}}};
    program.set_program_run_args(run_args);
    workload.update_program_run_args_for_device(top_row, second, run_args);
    const std::vector<uint32_t> before = WriterRuntimeArgsOnDeviceForTest(workload, second, NodeCoord{0, 0});
    program.set_program_run_args(run_args);
    EXPECT_EQ(WriterRuntimeArgsOnDeviceForTest(workload, second, NodeCoord{0, 0}), before);
}

TEST_F(ProgramSpecFabric1DMeshTest, CPU_TwoKernelsThatBindOneSpecGetEachDevicesConnectionArgs) {
    // On a Program on two devices, a second kernel that binds the spec on another node gets each device's args too.
    const distributed::MeshCoordinate second(0, 1);
    const distributed::MeshCoordinate below_second(1, 1);
    experimental::FabricConnectionsSpec connection =
        MakeFabricConnectionForTest(experimental::Fabric1DTargets{fixture_direction(second, below_second)});
    connection.links.insert({NodeCoord{1, 0}, 0});
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(connection);
    spec.advanced_options.fabric_connection_options.disable_exclusive_fabric_link_check = true;
    spec.kernels.push_back(test_helpers::MakeMinimalGen1DMKernel("reader"));
    spec.kernels[1].advanced_options.fabric_connection_bindings =
        spec.kernels[0].advanced_options.fabric_connection_bindings;
    spec.work_units.push_back(test_helpers::MakeMinimalWorkUnit("reader_unit", NodeCoord{1, 0}, {"reader"}));
    distributed::MeshWorkload workload(
        fixture_mesh_device(), {{distributed::MeshCoordinateRange({0, 0}, second), spec}});
    const uint32_t second_router =
        LegacyConnectionArgsForTest(fixture_mesh_device(), fixture_fabric_node_id(below_second), second)[0];
    EXPECT_THAT(
        KernelRuntimeArgsOnDeviceForTest(workload, "reader", second, NodeCoord{1, 0}),
        ElementsAre(second_router, _, _));
}

TEST_F(ProgramSpecFabric1DMeshTest, CPU_WraparoundRangeWithFabricConnectionsFails) {
    // A Program with fabric connections on several devices needs per-device run args, which a wraparound range lacks.
    const distributed::MeshCoordinate first(0, 0);
    const distributed::MeshCoordinate second(0, 1);
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(MakeFabricConnectionForTest(
        experimental::Fabric1DTargets{fixture_direction(first, distributed::MeshCoordinate(1, 0))}));
    const std::unordered_map<distributed::MeshCoordinateRange, ProgramSpec> specs = {
        {distributed::MeshCoordinateRange(second, first, fixture_mesh_device().shape()), spec}};
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_mesh_device(), specs); },
        ThrowsMessage<std::runtime_error>(HasSubstr("give each device its own Program")));
}

TEST_F(ProgramSpecFabric1DMeshTest, CPU_WraparoundRangeWithoutFabricConnectionsIsAccepted) {
    // A Program without fabric connections has no per-device run args, so it can run on a wraparound range.
    const distributed::MeshCoordinateRange wraparound(
        distributed::MeshCoordinate(0, 1), distributed::MeshCoordinate(0, 0), fixture_mesh_device().shape());
    const std::unordered_map<distributed::MeshCoordinateRange, ProgramSpec> specs = {
        {wraparound, test_helpers::MakeMinimalGen1ValidProgramSpec()}};
    EXPECT_NO_THROW(distributed::MeshWorkload(fixture_mesh_device(), specs));
}

TEST_F(ProgramSpecFabric1DMeshTest, CPU_ProgramOnOneDeviceHasNoPerDeviceRunArgs) {
    // A Program on one device holds its connection args in its own runtime args, without per-device run args.
    const distributed::MeshCoordinate second(0, 1);
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(MakeFabricConnectionForTest(
        experimental::Fabric1DTargets{fixture_direction(second, distributed::MeshCoordinate(1, 1))}));
    const distributed::MeshCoordinateRange one_device(second);
    distributed::MeshWorkload workload(
        fixture_mesh_device(), std::unordered_map<distributed::MeshCoordinateRange, ProgramSpec>{{one_device, spec}});
    EXPECT_EQ(PerDeviceRunArgsForTest(workload, one_device), nullptr);
}

}  // namespace
}  // namespace tt::tt_metal
