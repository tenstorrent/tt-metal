// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Local invariants of WorkUnitSpec::kernels (program_spec.hpp): at most one compute kernel, and summed
// num_threads within the per-architecture budgets.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <stdexcept>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen2ComputeKernel;
using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestQuasar;

TEST_F(ProgramSpecTestQuasar, CPU_WorkUnitWithMultipleComputeKernelsFails) {
    // A work_unit can have at most one compute kernel
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto compute1 = MakeMinimalGen2ComputeKernel("compute1");
    auto compute2 = MakeMinimalGen2ComputeKernel("compute2");

    spec.kernels = {compute1, compute2};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"compute1", "compute2"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("WorkUnitSpec 'work_unit' has more than one compute kernel")));
}

TEST_F(ProgramSpecTestQuasar, CPU_WorkUnitExceedsDMCoreBudgetFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    // Create enough DM kernels to exceed the 8 DM core budget
    auto kernel1 = MakeMinimalGen2DMKernel("dm1", 3);
    auto kernel2 = MakeMinimalGen2DMKernel("dm2", 3);
    auto kernel3 = MakeMinimalGen2DMKernel("dm3", 3);  // Total: 9 > 8

    spec.kernels = {kernel1, kernel2, kernel3};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm1", "dm2", "dm3"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("WorkUnitSpec 'work_unit' requests 9 data movement cores")));
}

TEST_F(ProgramSpecTestQuasar, CPU_WorkUnitExceedsComputeCoreBudgetFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    // Create enough compute kernels to exceed the 4 Tensix core budget (2+4=6).
    // (Legal thread counts on Quasar are 1, 2, 4; 3 is explicitly disallowed.)
    auto kernel1 = MakeMinimalGen2ComputeKernel("compute1", 2);
    auto kernel2 = MakeMinimalGen2ComputeKernel("compute2", 4);

    spec.kernels = {kernel1, kernel2};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"compute1", "compute2"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("WorkUnitSpec 'work_unit' needs 6 Tensix engines")));
}

TEST_F(ProgramSpecTestQuasar, CPU_MaxDMThreadsSucceeds) {
    // Use exactly 6 DM threads (the maximum available to the user)
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "max_dm_threads";

    auto producer = MakeMinimalGen2DMKernel("producer", 3);
    auto consumer = MakeMinimalGen2DMKernel("consumer", 3);  // Total: 6
    auto dfb = MakeMinimalDFB("dfb");
    dfb.num_entries = 9;  // must be a multiple of the number of threads

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
