// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ProgramSpec structural invariants (program_spec.hpp) of fabric connections: every target node runs exactly one
// kernel that binds the connection, and each (src, dst, link) appears at most once in the program.

#include <stdexcept>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestGen1;

// A fabric connection named `name` from chip 0 to chip 1 over `link_idx`, with one target on `node`.
FabricConnectionSpec MakeFabricConnectionForTest(const char* name, const NodeCoord& node, uint32_t link_idx = 0) {
    FabricConnectionSpec connection{
        .unique_id = FabricConnectionSpecName{name},
        .src_fabric_node_id = tt_fabric::FabricNodeId{tt_fabric::MeshId{0}, 0}};
    connection.targets[node] = {
        {.dst_fabric_node_id = tt_fabric::FabricNodeId{tt_fabric::MeshId{0}, 1}, .link_idx = link_idx}};
    return connection;
}

// A data-movement kernel named `name` that binds fabric connection `connection_name` as "fwd".
KernelSpec MakeKernelBindingFabricConnectionForTest(
    const char* name, const char* connection_name, DataMovementProcessor processor = DataMovementProcessor::RISCV_0) {
    KernelSpec kernel = MakeMinimalGen1DMKernel(name, processor);
    kernel.advanced_options.fabric_connection_bindings = {
        {.fabric_connection_spec_name = FabricConnectionSpecName{connection_name}, .accessor_name = "fwd"}};
    return kernel;
}

TEST_F(ProgramSpecTestGen1, CPU_FabricConnectionTargetNodeWithoutBindingKernelFails) {
    // Each target node runs a kernel that binds the connection.
    ProgramSpec spec{
        .kernels = {MakeKernelBindingFabricConnectionForTest("writer", "fwd")},
        .work_units = {MakeMinimalWorkUnit("work", NodeCoord{0, 0}, {"writer"})}};
    spec.advanced_options.fabric_connections = {MakeFabricConnectionForTest("fwd", NodeCoord{1, 0})};
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("no kernel that binds it runs there")));
}

TEST_F(ProgramSpecTestGen1, CPU_FabricConnectionBoundByTwoKernelsOnOneNodeFails) {
    // On each node, at most one kernel instance binds a fabric connection.
    ProgramSpec spec{
        .kernels =
            {MakeKernelBindingFabricConnectionForTest("writer", "fwd"),
             MakeKernelBindingFabricConnectionForTest("reader", "fwd", DataMovementProcessor::RISCV_1)},
        .work_units = {MakeMinimalWorkUnit("work", NodeCoord{0, 0}, {"writer", "reader"})}};
    spec.advanced_options.fabric_connections = {MakeFabricConnectionForTest("fwd", NodeCoord{0, 0})};
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("is bound by more than one kernel on node")));
}

TEST_F(ProgramSpecTestGen1, CPU_FabricConnectionSameLinkTwiceFails) {
    // Each (src, dst, link) appears at most once in a program: the router accepts one worker at a time.
    ProgramSpec spec{
        .kernels =
            {MakeKernelBindingFabricConnectionForTest("writer", "first"),
             MakeKernelBindingFabricConnectionForTest("other_writer", "second")},
        .work_units = {
            MakeMinimalWorkUnit("work", NodeCoord{0, 0}, {"writer"}),
            MakeMinimalWorkUnit("other_work", NodeCoord{1, 0}, {"other_writer"})}};
    spec.advanced_options.fabric_connections = {
        MakeFabricConnectionForTest("first", NodeCoord{0, 0}), MakeFabricConnectionForTest("second", NodeCoord{1, 0})};
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("more than once in the program")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
