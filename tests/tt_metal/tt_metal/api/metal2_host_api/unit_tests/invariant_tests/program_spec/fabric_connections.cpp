// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Structural invariants of ProgramAdvancedOptions::fabric_connections: who binds each spec, how the Program is created,
// and how the connections of the Program share the links of a device.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include <tt-metalium/workload/mesh_workload.hpp>
#include <tt-metalium/workload/program.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "metal2_host_api/test_helpers/fabric_mock_fixture.hpp"

namespace tt::tt_metal {
namespace {

using test_helpers::MakeFabricConnectionForTest;
using test_helpers::MakeSpecWithFabricConnectionForTest;
using test_helpers::ProgramSpecFabric1DTest;
using test_helpers::ProgramSpecFabric2DUDMTest;
using test_helpers::ProgramSpecFabricDisabledTest;
using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

TEST_F(ProgramSpecFabric1DTest, CPU_UnboundFabricConnectionFails) {
    // Each fabric connections spec is bound by a kernel.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(experimental::Fabric1DTargets{fixture_neighbour_direction()}));
    spec.kernels[0].advanced_options.fabric_connection_bindings.clear();
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_device(), spec); },
        ThrowsMessage<std::runtime_error>(HasSubstr("FabricConnectionsSpec 'conn' is defined but not bound")));
}

TEST_F(ProgramSpecFabric1DTest, CPU_FabricConnectionBoundByTwoKernelsOnOneNodeFails) {
    // On a node, at most one kernel binds a given fabric connections spec.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(experimental::Fabric1DTargets{fixture_neighbour_direction()}));
    KernelSpec reader = test_helpers::MakeMinimalGen1DMKernel("reader", DataMovementProcessor::RISCV_1);
    reader.advanced_options.fabric_connection_bindings = spec.kernels[0].advanced_options.fabric_connection_bindings;
    spec.kernels.push_back(reader);
    spec.work_units = {test_helpers::MakeMinimalWorkUnit("work_unit", NodeCoord{0, 0}, {"writer", "reader"})};
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_device(), spec); },
        ThrowsMessage<std::runtime_error>(HasSubstr("FabricConnectionsSpec 'conn' is bound by more than one kernel")));
}

TEST_F(ProgramSpecFabric1DTest, CPU_FabricConnectionLinkOnNodeWithoutBinderFails) {
    // Each node in links runs a kernel that binds the spec.
    experimental::FabricConnectionsSpec connection =
        MakeFabricConnectionForTest(experimental::Fabric1DTargets{fixture_neighbour_direction()});
    connection.links.insert({NodeCoord{1, 0}, 1});
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(connection);
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_device(), spec); },
        ThrowsMessage<std::runtime_error>(HasSubstr("no kernel that binds the spec runs there")));
}

TEST_F(ProgramSpecFabric1DTest, CPU_FabricConnectionBinderOnNodeOutsideLinksFails) {
    // Each node of a kernel that binds the spec is in links; the kernel is split otherwise.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(experimental::Fabric1DTargets{fixture_neighbour_direction()}));
    spec.work_units = {test_helpers::MakeMinimalWorkUnit("work_unit", NodeRange({0, 0}, {1, 0}), {"writer"})};
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_device(), spec); },
        ThrowsMessage<std::runtime_error>(::testing::AllOf(
            HasSubstr("Kernel 'writer' binds FabricConnectionsSpec 'conn', but node"),
            HasSubstr("Hint: split the KernelSpec"))));
}

TEST_F(ProgramSpecFabric1DTest, CPU_FabricConnectionLinkOutOfRangeFails) {
    // Each node's link exists in the resolved direction.
    experimental::FabricConnectionsSpec connection =
        MakeFabricConnectionForTest(experimental::Fabric1DTargets{fixture_neighbour_direction()});
    connection.links = {{NodeCoord{0, 0}, 1000}};
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(connection);
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_device(), spec); },
        ThrowsMessage<std::runtime_error>(HasSubstr("uses link 1000")));
}

TEST_F(ProgramSpecFabric1DTest, CPU_FabricConnectionOnNodeWithoutFreeSemaphoresFails) {
    // Each target of a node uses two of the semaphores of the node.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(experimental::Fabric1DTargets{fixture_neighbour_direction()}));
    for (int i = 0; i < 15; ++i) {
        spec.semaphores.push_back(
            SemaphoreSpec{.unique_id = SemaphoreSpecName{"sem" + std::to_string(i)}, .target_nodes = NodeCoord{0, 0}});
    }
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_device(), spec); },
        ThrowsMessage<std::runtime_error>(HasSubstr("has no semaphores left for the connection")));
}

TEST_F(ProgramSpecFabric1DTest, CPU_FabricConnectionUsesTheLastTwoFreeSemaphoresOfTheNode) {
    // A target can use the last two free semaphores of a node.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(experimental::Fabric1DTargets{fixture_neighbour_direction()}));
    for (int i = 0; i < 14; ++i) {
        spec.semaphores.push_back(
            SemaphoreSpec{.unique_id = SemaphoreSpecName{"sem" + std::to_string(i)}, .target_nodes = NodeCoord{0, 0}});
    }
    EXPECT_NO_THROW(distributed::MeshWorkload(fixture_device(), spec));
}

TEST_F(ProgramSpecFabric1DTest, CPU_TwoFabricConnectionsSpecsWithOneNameFail) {
    // Each FabricConnectionsSpec of a ProgramSpec has its own name.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(experimental::Fabric1DTargets{fixture_neighbour_direction()}));
    spec.advanced_options.fabric_connections.push_back(
        MakeFabricConnectionForTest(experimental::Fabric1DTargets{fixture_other_neighbour_direction()}));
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_device(), spec); },
        ThrowsMessage<std::runtime_error>(HasSubstr("Duplicate FabricConnectionsSpec name 'conn'")));
}

TEST_F(ProgramSpecFabric1DTest, CPU_FabricConnectionToZWithoutNeighbourFailsOnlyForTheMissingNeighbour) {
    // Z is an allowed direction target; it fails only on a device without a fabric neighbour in Z.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(experimental::Fabric1DTargets{tt_fabric::RoutingDirection::Z}));
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_device(), spec); },
        ThrowsMessage<std::runtime_error>(HasSubstr("has no fabric neighbour in direction")));
}

TEST_F(ProgramSpecFabricDisabledTest, CPU_FabricConnectionWithFabricDisabledFails) {
    // A Program with fabric connections needs the fabric to be enabled.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(experimental::Fabric1DTargets{tt_fabric::RoutingDirection::E}));
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_device(), spec); },
        ThrowsMessage<std::runtime_error>(HasSubstr("needs the fabric to be enabled")));
}

TEST_F(ProgramSpecFabricDisabledTest, CPU_ProgramWithoutFabricConnectionsWithFabricDisabledIsAccepted) {
    // A Program without fabric connections does not need the fabric.
    ProgramSpec spec = test_helpers::MakeMinimalGen1ValidProgramSpec();
    EXPECT_NO_THROW(distributed::MeshWorkload(fixture_device(), spec));
}

TEST_F(ProgramSpecFabric1DTest, CPU_TwoFabricConnectionsOnOneLinkFail) {
    // On a device, each (direction, link) is used by at most one connection of the Program.
    experimental::FabricConnectionsSpec second =
        MakeFabricConnectionForTest(experimental::Fabric1DTargets{fixture_neighbour_direction()});
    second.unique_id = experimental::FabricConnectionsSpecName{"second"};
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(experimental::Fabric1DTargets{fixture_neighbour_direction()}));
    spec.advanced_options.fabric_connections.push_back(second);
    spec.kernels[0].advanced_options.fabric_connection_bindings.push_back(experimental::FabricConnectionBinding{
        .fabric_connections_spec_name = second.unique_id, .accessor_name = "second"});
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_device(), spec); },
        ThrowsMessage<std::runtime_error>(HasSubstr("is used by more than one fabric connection")));
}

TEST_F(ProgramSpecFabric1DTest, CPU_TwoFabricConnectionsOnOneLinkWithCheckDisabledAreAccepted) {
    // disable_exclusive_fabric_link_check lets two connections of the Program share a link.
    experimental::FabricConnectionsSpec second =
        MakeFabricConnectionForTest(experimental::Fabric1DTargets{fixture_neighbour_direction()});
    second.unique_id = experimental::FabricConnectionsSpecName{"second"};
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(experimental::Fabric1DTargets{fixture_neighbour_direction()}));
    spec.advanced_options.fabric_connections.push_back(second);
    spec.kernels[0].advanced_options.fabric_connection_bindings.push_back(experimental::FabricConnectionBinding{
        .fabric_connections_spec_name = second.unique_id, .accessor_name = "second"});
    spec.advanced_options.fabric_connection_options.disable_exclusive_fabric_link_check = true;
    EXPECT_NO_THROW(distributed::MeshWorkload(fixture_device(), spec));
}

TEST_F(ProgramSpecFabric2DUDMTest, CPU_FabricConnectionOnUDMFabricFails) {
    // Fabric connections do not support the UDM fabric tensix config.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(experimental::Fabric2DTargets{fixture_neighbour_fabric_node_id()}));
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_device(), spec); },
        ThrowsMessage<std::runtime_error>(HasSubstr("does not support the UDM fabric tensix config")));
}

}  // namespace
}  // namespace tt::tt_metal
