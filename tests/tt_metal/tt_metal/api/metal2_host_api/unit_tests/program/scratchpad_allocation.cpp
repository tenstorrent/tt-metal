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

}  // namespace
}  // namespace tt::tt_metal::experimental
