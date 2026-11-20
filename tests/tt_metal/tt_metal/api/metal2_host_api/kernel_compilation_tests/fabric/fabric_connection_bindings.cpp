// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// JIT-compile tests for fabric connection binding tokens: a kernel that binds a FabricConnectionSpec gets a
// fabric::<accessor_name> token with one slot for each target (mock 2-chip Wormhole).

#include <gtest/gtest.h>

#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "metal2_host_api/test_helpers/fabric_mock_fixture.hpp"

namespace tt::tt_metal {
namespace {

using test_helpers::MakeFabricConnectionForTest;
using test_helpers::MakeSpecWithFabricConnectionForTest;
using test_helpers::ProgramSpecFabric1DTest;

constexpr const char* kTwoTargetTokenKernel = R"(
#include "api/dataflow/fabric_connection_binding_token.h"
static_assert(fabric::conn.num_targets() == 2, "fabric::conn has one slot per target");
void kernel_main() {}
)";

TEST_F(ProgramSpecFabric1DTest, CPU_FabricConnectionTokenHasOneSlotPerTargetJITSmoke) {
    // The binding token of a two-target spec has two slots.
    ProgramSpec spec = MakeSpecWithFabricConnectionForTest(MakeFabricConnectionForTest(
        Fabric1DTargets{fixture_neighbour_direction(), fixture_direction_without_neighbour()}));
    spec.kernels[0].source = KernelSpec::SourceCode{kTwoTargetTokenKernel};
    EXPECT_NO_THROW(distributed::MeshWorkload(fixture_device(), spec));
}

}  // namespace
}  // namespace tt::tt_metal
