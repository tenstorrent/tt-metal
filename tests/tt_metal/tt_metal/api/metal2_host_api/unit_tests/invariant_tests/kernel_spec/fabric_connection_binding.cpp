// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Invariants of KernelSpec::fabric_connection_bindings: references, accessor names and kernel type.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <stdexcept>

#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "metal2_host_api/test_helpers/fabric_mock_fixture.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal {
namespace {

using test_helpers::MakeFabricConnectionForTest;
using test_helpers::MakeSpecWithFabricConnectionForTest;
using test_helpers::ProgramSpecFabric1DTest;
using test_helpers::ProgramSpecTestQuasar;
using ::testing::HasSubstr;
using ::testing::ThrowsMessage;

TEST_F(ProgramSpecFabric1DTest, CPU_FabricConnectionBindingToUnknownSpecFails) {
    // A binding names a FabricConnectionSpec of the ProgramSpec.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(Fabric1DTargets{fixture_neighbour_direction()}));
    spec.kernels[0].fabric_connection_bindings.push_back(FabricConnectionBinding{
        .fabric_connection_spec_name = FabricConnectionSpecName{"missing"}, .accessor_name = "x"});
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_device(), spec); },
        ThrowsMessage<std::runtime_error>(HasSubstr("references unknown FabricConnectionSpec 'missing'")));
}

TEST_F(ProgramSpecFabric1DTest, CPU_FabricConnectionBindingWithRepeatedAccessorNameFails) {
    // Each fabric connection binding of a kernel has its own accessor name.
    FabricConnectionSpec second = MakeFabricConnectionForTest(Fabric1DTargets{fixture_direction_without_neighbour()});
    second.unique_id = FabricConnectionSpecName{"second"};
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(Fabric1DTargets{fixture_neighbour_direction()}));
    spec.fabric_connections.push_back(second);
    spec.kernels[0].fabric_connection_bindings.push_back(
        FabricConnectionBinding{.fabric_connection_spec_name = second.unique_id, .accessor_name = "conn"});
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_device(), spec); },
        ThrowsMessage<std::runtime_error>(HasSubstr("duplicate fabric connection accessor_name 'conn'")));
}

TEST_F(ProgramSpecFabric1DTest, CPU_FabricConnectionBoundTwiceByOneKernelFails) {
    // A kernel binds each FabricConnectionSpec at most once.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(Fabric1DTargets{fixture_neighbour_direction()}));
    spec.kernels[0].fabric_connection_bindings.push_back(FabricConnectionBinding{
        .fabric_connection_spec_name = FabricConnectionSpecName{"conn"}, .accessor_name = "conn_again"});
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_device(), spec); },
        ThrowsMessage<std::runtime_error>(
            HasSubstr("Kernel 'writer' binds FabricConnectionSpec 'conn' more than once")));
}

TEST_F(ProgramSpecFabric1DTest, CPU_FabricConnectionBindingWithInvalidAccessorNameFails) {
    // An accessor name is a C++ identifier.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(Fabric1DTargets{fixture_neighbour_direction()}));
    spec.kernels[0].fabric_connection_bindings[0].accessor_name = "not valid";
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_device(), spec); },
        ThrowsMessage<std::runtime_error>(HasSubstr("must be a valid C++ identifier")));
}

TEST_F(ProgramSpecFabric1DTest, CPU_FabricConnectionBoundByComputeKernelFails) {
    // Only data-movement kernels bind fabric connections.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(Fabric1DTargets{fixture_neighbour_direction()}));
    KernelSpec compute = test_helpers::MakeMinimalGen1ComputeKernel("compute");
    compute.fabric_connection_bindings = spec.kernels[0].fabric_connection_bindings;
    spec.kernels = {compute};
    spec.work_units = {test_helpers::MakeMinimalWorkUnit("work_unit", NodeCoord{0, 0}, {"compute"})};
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(fixture_device(), spec); },
        ThrowsMessage<std::runtime_error>(HasSubstr("only data-movement kernels can bind a FabricConnectionSpec")));
}

TEST_F(ProgramSpecTestQuasar, CPU_FabricConnectionBoundByGen2KernelFails) {
    // Fabric connection bindings are not supported on 2nd-gen architectures yet.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        MakeFabricConnectionForTest(Fabric1DTargets{tt_fabric::RoutingDirection::E}));
    KernelSpec writer = test_helpers::MakeMinimalGen2DMKernel("writer");
    writer.fabric_connection_bindings = spec.kernels[0].fabric_connection_bindings;
    spec.kernels = {writer};
    EXPECT_THAT(
        [&] { distributed::MeshWorkload(*mesh_device_, spec); },
        ThrowsMessage<std::runtime_error>(HasSubstr("not supported on 2nd-gen architectures")));
}

}  // namespace
}  // namespace tt::tt_metal
