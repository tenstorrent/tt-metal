// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Where MakeProgramFromSpec places a scratchpad in L1 relative to the program's dataflow buffers.

#include <gtest/gtest.h>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestQuasar;

const ScratchpadSpec kScratchpad{.unique_id = ScratchpadSpecName{"scratch_0"}, .size_per_node = 1024};
const KernelSpec::ScratchpadBinding kScratchpadBinding{
    .scratchpad_spec_name = ScratchpadSpecName{"scratch_0"}, .accessor_name = "s"};

// A scratchpad on the same node as a DFB starts where the DFB ends.
TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadStacksOnDataflowBuffer) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();  // dfb_0 (1024 * 2) on node (0,0)
    spec.scratchpads = {kScratchpad};
    spec.kernels[0].scratchpad_bindings = {kScratchpadBinding};
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    const auto& scratchpads = program.impl().get_kernel_by_spec_name("dm_kernel")->scratchpad_binding_handles();
    ASSERT_EQ(scratchpads.size(), 1u);
    EXPECT_EQ(scratchpads[0].allocated_address, dfb->uniform_alloc_addr() + dfb->total_size());
}

// A DFB on a different node does not move the scratchpad: both start at the bottom of program L1.
TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadIgnoresDataflowBufferOnOtherNode) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();  // dfb_0 on node (0,0)
    auto scratch_kernel = MakeMinimalGen2DMKernel("scratch_kernel");
    scratch_kernel.scratchpad_bindings = {kScratchpadBinding};
    spec.kernels.push_back(scratch_kernel);
    spec.scratchpads = {kScratchpad};
    spec.work_units.push_back(MakeMinimalWorkUnit("work_unit_1", NodeCoord{1, 0}, {"scratch_kernel"}));
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    const auto& scratchpads = program.impl().get_kernel_by_spec_name("scratch_kernel")->scratchpad_binding_handles();
    ASSERT_EQ(scratchpads.size(), 1u);
    EXPECT_EQ(scratchpads[0].allocated_address, dfb->uniform_alloc_addr());
}

// A DFB on another node does not push the scratchpad up, even when it shares a core range with a DFB the
// scratchpad does sit on. Nodes L = (0,0) and R = (1,0):
//   wide_dfb on L and R, then left_dfb on L only, then a scratchpad on R only.
// The scratchpad belongs directly on top of wide_dfb; left_dfb is not on R.
TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadIgnoresDataflowBufferOnOtherNodeOfSharedRange) {
    const NodeCoord left{0, 0};
    const NodeCoord right{1, 0};

    auto wide_producer = MakeMinimalGen2DMKernel("wide_producer");
    auto wide_consumer = MakeMinimalGen2DMKernel("wide_consumer");
    wide_producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"wide_dfb"}, "out"));
    wide_consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"wide_dfb"}, "in"));

    auto left_producer = MakeMinimalGen2DMKernel("left_producer");
    auto left_consumer = MakeMinimalGen2DMKernel("left_consumer");
    left_producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"left_dfb"}, "out"));
    left_consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"left_dfb"}, "in"));

    auto scratch_kernel = MakeMinimalGen2DMKernel("scratch_kernel");
    scratch_kernel.scratchpad_bindings = {kScratchpadBinding};

    ProgramSpec spec;
    spec.name = "scratchpad_shared_range";
    spec.kernels = {wide_producer, wide_consumer, left_producer, left_consumer, scratch_kernel};
    spec.dataflow_buffers = {
        MakeMinimalDFB("wide_dfb", /*entry_size=*/1024, /*num_entries=*/2),
        MakeMinimalDFB("left_dfb", /*entry_size=*/1024, /*num_entries=*/4),
    };
    spec.scratchpads = {kScratchpad};
    spec.work_units = {
        MakeMinimalWorkUnit("left", left, {"wide_producer", "wide_consumer", "left_producer", "left_consumer"}),
        MakeMinimalWorkUnit("right", right, {"wide_producer", "wide_consumer", "scratch_kernel"}),
    };
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    auto wide_dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("wide_dfb"));
    auto left_dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("left_dfb"));
    // Guards against passing vacuously: the case needs wide_dfb on one core range spanning both nodes.
    ASSERT_EQ(wide_dfb->core_ranges.ranges().size(), 1u);
    ASSERT_EQ(left_dfb->uniform_alloc_addr(), wide_dfb->uniform_alloc_addr() + wide_dfb->total_size());

    const auto& scratchpads = program.impl().get_kernel_by_spec_name("scratch_kernel")->scratchpad_binding_handles();
    ASSERT_EQ(scratchpads.size(), 1u);
    EXPECT_EQ(scratchpads[0].allocated_address, wide_dfb->uniform_alloc_addr() + wide_dfb->total_size());
}

}  // namespace
}  // namespace tt::tt_metal::experimental
