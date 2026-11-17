// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdexcept>

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

TEST_F(ProgramSpecTestGen1, CPU_FabricConnectionNodeWithTwoTargetsFails) {
    // A node of a fabric connection has at most one (dst, link) target.
    const tt_fabric::FabricNodeId src{tt_fabric::MeshId{0}, 0};
    const tt_fabric::FabricNodeId dst{tt_fabric::MeshId{0}, 1};
    FabricConnectionSpec connection{.unique_id = FabricConnectionSpecName{"fwd"}, .src_fabric_node_id = src};
    connection.targets[NodeCoord{0, 0}] = {
        {.dst_fabric_node_id = dst, .link_idx = 0}, {.dst_fabric_node_id = dst, .link_idx = 1}};
    ProgramSpec spec{
        .kernels = {MakeMinimalGen1DMKernel("writer")},
        .work_units = {MakeMinimalWorkUnit("work", NodeCoord{0, 0}, {"writer"})}};
    spec.kernels[0].advanced_options.fabric_connection_bindings = {
        {.fabric_connection_spec_name = FabricConnectionSpecName{"fwd"}, .accessor_name = "fwd"}};
    spec.advanced_options.fabric_connections = {connection};
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("a node has at most one target")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
