// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Local invariants of WorkUnitSpec (program_spec.hpp): non-empty kernels, target nodes on the device grid.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <set>
#include <stdexcept>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "metal2_host_api/test_helpers.hpp"
#include "metal2_host_api/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestGen1;
using test_helpers::ProgramSpecTestQuasar;

TEST_F(ProgramSpecTestQuasar, CPU_WorkUnitSpecWithNoKernelsFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel = MakeMinimalGen2DMKernel("kernel");
    spec.kernels = {kernel};

    WorkUnitSpec work_unit;
    work_unit.name = "work_unit";
    work_unit.target_nodes = node;
    work_unit.kernels = {};  // No kernels!
    spec.work_units = std::vector<WorkUnitSpec>{work_unit};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'kernel' is not referenced by any WorkUnitSpec")));
}

TEST_F(ProgramSpecTestQuasar, CPU_NodeRangeSetTargetNodesSucceeds) {
    // Test with NodeRangeSet (multiple disjoint ranges)
    NodeRangeSet nodes(std::set<NodeRange>{NodeRange{{0, 0}, {0, 1}}, NodeRange{{2, 0}, {2, 1}}});

    ProgramSpec spec;
    spec.name = "range_set_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer = MakeMinimalGen2DMKernel("consumer");
    auto dfb = MakeMinimalDFB("dfb");

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", nodes, {"producer", "consumer"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_MultiNodeProgramSucceeds) {
    // A program spanning multiple nodes
    NodeRange nodes{{0, 0}, {1, 1}};  // 2x2 grid

    ProgramSpec spec;
    spec.name = "multi_node_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer = MakeMinimalGen2DMKernel("consumer");
    auto dfb = MakeMinimalDFB("dfb");

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", nodes, {"producer", "consumer"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

// WH N150 mock grid reference (wormhole_N150.yaml, harvest_mask=0x40 = 1 row harvested):
//   - Fast dispatch: compute_grid = 8x8 (y in [0,7]; one row reserved for dispatch)
//   - Slow dispatch: compute_grid = 8x9 (y in [0,8]; full logical tensix grid, no rows reserved)
//
// The apparent grid size is different in slow dispatch vs. fast dispatch mode. CI runs with
// both, so choose OOB coordinates that will fail in both cases.
//
// These tests use the WH mock device, not real hardware.

TEST_F(ProgramSpecTestGen1, CPU_KernelTargetsNodeBeyondGridYFails) {
    // y=9 is just outside the 9-row slow-dispatch grid (also outside the 8-row fast-dispatch grid).
    const NodeCoord oob_node{0, 9};

    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel = MakeMinimalGen1DMKernel("dm_kernel");
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", oob_node, {"dm_kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("out of bounds")));
}

TEST_F(ProgramSpecTestGen1, CPU_KernelTargetsOutOfBoundsNodeFails) {
    // x=8 is just outside the 8-column grid (same in fast and slow dispatch).
    const NodeCoord oob_node{8, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel = MakeMinimalGen1DMKernel("dm_kernel");
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", oob_node, {"dm_kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("out of bounds")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
