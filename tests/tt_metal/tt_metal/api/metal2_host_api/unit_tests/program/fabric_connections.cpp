// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fabric connections (ProgramAdvancedOptions::fabric_connections): what a bound kernel receives per node.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/fabric_mock_fixture.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecFabricTestGen1;

// A data-movement kernel "writer" on `kernel_nodes` binds fabric connection "fwd" as "fwd".
// The connection goes from `src` to `dst` over link 0, from `target_node` only.
ProgramSpec MakeSpecWithFabricConnectionForTest(
    const tt_fabric::FabricNodeId& src,
    const tt_fabric::FabricNodeId& dst,
    const NodeRangeSet& kernel_nodes,
    const NodeCoord& target_node) {
    KernelSpec writer = MakeMinimalGen1DMKernel("writer");
    writer.advanced_options.fabric_connection_bindings = {
        {.fabric_connection_spec_name = FabricConnectionSpecName{"fwd"}, .accessor_name = "fwd"}};
    FabricConnectionSpec connection{.unique_id = FabricConnectionSpecName{"fwd"}, .src_fabric_node_id = src};
    connection.targets[target_node] = {{.dst_fabric_node_id = dst, .link_idx = 0}};
    ProgramSpec spec{.kernels = {writer}, .work_units = {MakeMinimalWorkUnit("work", kernel_nodes, {"writer"})}};
    spec.advanced_options.fabric_connections = {connection};
    return spec;
}

// Not a public API: the runtime args of `kernel_name` on `node`, as the device will read them.
std::vector<uint32_t> RuntimeArgsOfKernelOnNode(
    Program& program, const std::string& kernel_name, const NodeCoord& node) {
    return program.impl().get_kernel_by_spec_name(kernel_name)->runtime_args(node);
}

// The connection args that the legacy append_fabric_connection_rt_args gives a worker on `node` of a new Program.
std::vector<uint32_t> LegacyFabricConnectionArgsForTest(
    distributed::MeshDevice& device,
    const tt_fabric::FabricNodeId& src,
    const tt_fabric::FabricNodeId& dst,
    const NodeCoord& node) {
    Program legacy_program = CreateProgram();
    std::vector<uint32_t> args;
    tt_fabric::append_fabric_connection_rt_args(device, src, dst, /*link_idx=*/0, legacy_program, node, args);
    return args;
}

TEST_F(ProgramSpecFabricTestGen1, CPU_FabricConnectionArgsMatchLegacyConnectionArgs) {
    // A connected node gets the same connection args as append_fabric_connection_rt_args, after a "connected" word.
    const tt_fabric::FabricNodeId src = fixture_source_fabric_node_id();
    const tt_fabric::FabricNodeId dst = fixture_destination_fabric_node_id();
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(src, dst, NodeRangeSet(NodeRange({0, 0})), NodeCoord{0, 0});
    Program program = MakeProgramFromSpec(fixture_source_device(), spec);
    SetProgramRunArgs(program, {});
    std::vector<uint32_t> expected = {1u};
    for (uint32_t word : LegacyFabricConnectionArgsForTest(fixture_source_device(), src, dst, NodeCoord{0, 0})) {
        expected.push_back(word);
    }
    EXPECT_EQ(RuntimeArgsOfKernelOnNode(program, "writer", NodeCoord{0, 0}), expected);
}

TEST_F(ProgramSpecFabricTestGen1, CPU_FabricConnectionNodeWithoutTargetIsNotConnected) {
    // A node of the binding kernel without a target gets no connection.
    const NodeRangeSet kernel_nodes(NodeRange({0, 0}, {1, 0}));
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(
        fixture_source_fabric_node_id(), fixture_destination_fabric_node_id(), kernel_nodes, NodeCoord{0, 0});
    Program program = MakeProgramFromSpec(fixture_source_device(), spec);
    SetProgramRunArgs(program, {});
    EXPECT_THAT(
        RuntimeArgsOfKernelOnNode(program, "writer", NodeCoord{1, 0}),
        ::testing::AllOf(::testing::Not(::testing::IsEmpty()), ::testing::Each(0u)));
}

TEST_F(ProgramSpecFabricTestGen1, CPU_FabricConnectionToItsOwnDeviceFails) {
    // A fabric connection goes to another fabric node.
    const tt_fabric::FabricNodeId src = fixture_source_fabric_node_id();
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(src, src, NodeRangeSet(NodeRange({0, 0})), NodeCoord{0, 0});
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(fixture_source_device(), spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Expected different src and dst")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
