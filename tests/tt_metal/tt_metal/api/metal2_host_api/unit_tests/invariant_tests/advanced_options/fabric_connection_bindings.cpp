// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Local invariants of KernelAdvancedOptions::fabric_connection_bindings.

#include <stdexcept>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalGen1ComputeKernel;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestGen1;
using test_helpers::ProgramSpecTestQuasar;

// `kernel` on node {0, 0}, and a fabric connection "fwd" with one target on that node.
ProgramSpec MakeSpecWithFabricConnectionForTest(KernelSpec kernel) {
    FabricConnectionSpec connection{
        .unique_id = FabricConnectionSpecName{"fwd"},
        .src_fabric_node_id = tt_fabric::FabricNodeId{tt_fabric::MeshId{0}, 0}};
    connection.targets[NodeCoord{0, 0}] = {
        {.dst_fabric_node_id = tt_fabric::FabricNodeId{tt_fabric::MeshId{0}, 1}, .link_idx = 0}};
    ProgramSpec spec;
    spec.kernels = {std::move(kernel)};
    spec.work_units = {MakeMinimalWorkUnit("work", NodeCoord{0, 0}, {spec.kernels[0].unique_id.get()})};
    spec.advanced_options.fabric_connections = {connection};
    return spec;
}

TEST_F(ProgramSpecTestGen1, CPU_FabricConnectionBindingOnComputeKernelFails) {
    // A compute kernel cannot bind a fabric connection.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(MakeMinimalGen1ComputeKernel("compute"));
    spec.kernels[0].advanced_options.fabric_connection_bindings = {
        {.fabric_connection_spec_name = FabricConnectionSpecName{"fwd"}, .accessor_name = "fwd"}};
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("compute kernel cannot bind")));
}

TEST_F(ProgramSpecTestGen1, CPU_FabricConnectionBindingDuplicateAccessorNameFails) {
    // Two fabric connection bindings of one kernel cannot share an accessor name.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(MakeMinimalGen1DMKernel("writer"));
    spec.kernels[0].advanced_options.fabric_connection_bindings = {
        {.fabric_connection_spec_name = FabricConnectionSpecName{"fwd"}, .accessor_name = "fwd"},
        {.fabric_connection_spec_name = FabricConnectionSpecName{"fwd"}, .accessor_name = "fwd"}};
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("duplicate fabric connection accessor_name 'fwd'")));
}

TEST_F(ProgramSpecTestGen1, CPU_FabricConnectionBindingInvalidAccessorNameFails) {
    // A fabric connection accessor name must be a valid C++ identifier.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(MakeMinimalGen1DMKernel("writer"));
    spec.kernels[0].advanced_options.fabric_connection_bindings = {
        {.fabric_connection_spec_name = FabricConnectionSpecName{"fwd"}, .accessor_name = "not-an-identifier"}};
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("must be a valid C++ identifier")));
}

TEST_F(ProgramSpecTestQuasar, CPU_FabricConnectionBindingOnMultiThreadKernelFails) {
    // A kernel with more than one thread cannot bind a fabric connection.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(MakeMinimalGen2DMKernel("writer", /*num_threads=*/2));
    spec.kernels[0].advanced_options.fabric_connection_bindings = {
        {.fabric_connection_spec_name = FabricConnectionSpecName{"fwd"}, .accessor_name = "fwd"}};
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("must have num_threads == 1")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
