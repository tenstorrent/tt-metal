// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Structural invariants of ProgramSpec::fabric_connections: who binds each spec, how the Program is created,
// and how the connections of the Program share the links of a device.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "metal2_host_api/test_helpers/fabric_mock_fixture.hpp"

namespace tt::tt_metal {
namespace {

using test_helpers::MakeFabricConnectionForTest;
using test_helpers::MakeSpecWithFabricConnectionForTest;
using test_helpers::ProgramSpecFabric1DTest;
using test_helpers::ProgramSpecFabric2DUDMTest;
using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

TEST_F(ProgramSpecFabric1DTest, CPU_ProgramConstructorRejectsFabricConnections) {
    // A Program with fabric connections is created by a MeshWorkload constructor, which knows its device.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(Fabric1DTargets{fixture_neighbour_direction()}));
    EXPECT_THAT(
        [&] { Program(fixture_device(), spec); },
        ThrowsMessage<std::runtime_error>(HasSubstr("only a MeshWorkload constructor can create")));
}

TEST_F(ProgramSpecFabric1DTest, CPU_UnboundFabricConnectionFails) {
    // Each fabric connection spec is bound by a kernel.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(Fabric1DTargets{fixture_neighbour_direction()}));
    spec.kernels[0].fabric_connection_bindings.clear();
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_device(), spec); },
        ThrowsMessage<std::runtime_error>(HasSubstr("FabricConnectionSpec 'conn' is defined but not bound")));
}

TEST_F(ProgramSpecFabric1DTest, CPU_FabricConnectionBoundByTwoKernelsOnOneNodeFails) {
    // On a node, at most one kernel binds a given fabric connection spec.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(Fabric1DTargets{fixture_neighbour_direction()}));
    KernelSpec reader = test_helpers::MakeMinimalGen1DMKernel("reader", DataMovementProcessor::RISCV_1);
    reader.fabric_connection_bindings = spec.kernels[0].fabric_connection_bindings;
    spec.kernels.push_back(reader);
    spec.work_units = {test_helpers::MakeMinimalWorkUnit("work_unit", NodeCoord{0, 0}, {"writer", "reader"})};
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_device(), spec); },
        ThrowsMessage<std::runtime_error>(HasSubstr("FabricConnectionSpec 'conn' is bound by more than one kernel")));
}

TEST_F(ProgramSpecFabric1DTest, CPU_FabricConnectionLinkOnNodeWithoutBinderFails) {
    // Each node in links runs a kernel that binds the spec.
    FabricConnectionSpec connection = MakeFabricConnectionForTest(Fabric1DTargets{fixture_neighbour_direction()});
    connection.links.insert({NodeCoord{1, 0}, 1});
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(connection);
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_device(), spec); },
        ThrowsMessage<std::runtime_error>(HasSubstr("no kernel that binds the spec runs there")));
}

TEST_F(ProgramSpecFabric1DTest, CPU_FabricConnectionLinkOutOfRangeFails) {
    // Each node's link exists in the resolved direction.
    FabricConnectionSpec connection = MakeFabricConnectionForTest(Fabric1DTargets{fixture_neighbour_direction()});
    connection.links = {{NodeCoord{0, 0}, 1000}};
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(connection);
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_device(), spec); },
        ThrowsMessage<std::runtime_error>(HasSubstr("uses link 1000")));
}

TEST_F(ProgramSpecFabric1DTest, CPU_FabricConnectionOnNodeWithoutFreeSemaphoresFails) {
    // Each connected target of a node uses two of the semaphores of the node.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(Fabric1DTargets{fixture_neighbour_direction()}));
    for (int i = 0; i < 15; ++i) {
        spec.semaphores.push_back(
            SemaphoreSpec{.unique_id = SemaphoreSpecName{"sem" + std::to_string(i)}, .target_nodes = NodeCoord{0, 0}});
    }
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_device(), spec); },
        ThrowsMessage<std::runtime_error>(HasSubstr("has no semaphores left for the connection")));
}

TEST_F(ProgramSpecFabric1DTest, CPU_TwoFabricConnectionsOnOneLinkFail) {
    // On a device, each (direction, link) is used by at most one connection of the Program.
    FabricConnectionSpec second = MakeFabricConnectionForTest(Fabric1DTargets{fixture_neighbour_direction()});
    second.unique_id = FabricConnectionSpecName{"second"};
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(Fabric1DTargets{fixture_neighbour_direction()}));
    spec.fabric_connections.push_back(second);
    spec.kernels[0].fabric_connection_bindings.push_back(
        FabricConnectionBinding{.fabric_connection_spec_name = second.unique_id, .accessor_name = "second"});
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_device(), spec); },
        ThrowsMessage<std::runtime_error>(HasSubstr("is used by more than one fabric connection")));
}

TEST_F(ProgramSpecFabric1DTest, CPU_TwoFabricConnectionsOnOneLinkWithCheckDisabledAreAccepted) {
    // disable_exclusive_fabric_link_check lets two connections of the Program share a link.
    FabricConnectionSpec second = MakeFabricConnectionForTest(Fabric1DTargets{fixture_neighbour_direction()});
    second.unique_id = FabricConnectionSpecName{"second"};
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(Fabric1DTargets{fixture_neighbour_direction()}));
    spec.fabric_connections.push_back(second);
    spec.kernels[0].fabric_connection_bindings.push_back(
        FabricConnectionBinding{.fabric_connection_spec_name = second.unique_id, .accessor_name = "second"});
    spec.advanced_options.fabric_connection_options.disable_exclusive_fabric_link_check = true;
    EXPECT_NO_THROW(distributed::MeshWorkload(fixture_device(), spec));
}

TEST_F(ProgramSpecFabric2DUDMTest, CPU_FabricConnectionOnUDMFabricFails) {
    // Fabric connections do not support the UDM fabric tensix config.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(Fabric2DTargets{fixture_neighbour_fabric_node_id()}));
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_device(), spec); },
        ThrowsMessage<std::runtime_error>(HasSubstr("does not support the UDM fabric tensix config")));
}

}  // namespace
}  // namespace tt::tt_metal
