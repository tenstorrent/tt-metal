// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Per-field invariants of ProgramSpec (program_spec.hpp): unique ids within each group, non-empty
// kernels and work_units, disjoint work_units, no cross-node DFBs yet.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <set>
#include <stdexcept>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindTensorParameterToKernel;
using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalTensorParameter;
using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestGen1;
using test_helpers::ProgramSpecTestQuasar;

TEST_F(ProgramSpecTestQuasar, CPU_DuplicateKernelNameFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Add a kernel with duplicate name
    auto duplicate_kernel = MakeMinimalGen2DMKernel("dm_kernel");
    duplicate_kernel.hw_config = DataMovementHardwareConfig{};
    spec.kernels.push_back(duplicate_kernel);

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Duplicate KernelSpec name 'dm_kernel'")));
}

TEST_F(ProgramSpecTestGen1, CPU_DuplicateKernelNameFails) {
    // Structural validation (CollectSpecData) must catch this on gen1 too
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto k0 = MakeMinimalGen1DMKernel("dm_kernel", DataMovementProcessor::RISCV_0);
    auto k1 = MakeMinimalGen1DMKernel("dm_kernel", DataMovementProcessor::RISCV_1);  // duplicate name

    spec.kernels = {k0, k1};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"dm_kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Duplicate KernelSpec name")));
}

TEST_F(ProgramSpecTestQuasar, CPU_DuplicateDFBNameFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Add a DFB with duplicate name
    auto duplicate_dfb = MakeMinimalDFB("dfb_0");
    spec.dataflow_buffers.push_back(duplicate_dfb);

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Duplicate DataflowBufferSpec name 'dfb_0'")));
}

TEST_F(ProgramSpecTestQuasar, CPU_DuplicateSemaphoreNameFails) {
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    // Add two semaphores with the same name
    SemaphoreSpec sem1;
    sem1.unique_id = SemaphoreSpecName{"sem_0"};
    sem1.target_nodes = NodeCoord{0, 0};

    SemaphoreSpec sem2;
    sem2.unique_id = SemaphoreSpecName{"sem_0"};  // duplicate!
    sem2.target_nodes = NodeCoord{1, 0};

    spec.semaphores = {sem1, sem2};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Duplicate SemaphoreSpec name 'sem_0'")));
}

TEST_F(ProgramSpecTestQuasar, CPU_DuplicateScratchpadNameFails) {
    // Two ScratchpadSpecs declared with the same unique_id.
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    spec.scratchpads = {
        ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch_0"}, .size_per_node = 1024},
        ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch_0"}, .size_per_node = 512},  // duplicate!
    };
    spec.kernels[0].scratchpad_bindings = {
        KernelSpec::ScratchpadBinding{.scratchpad_spec_name = ScratchpadSpecName{"scratch_0"}, .accessor_name = "s"}};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("Duplicate ScratchpadSpec name")));
}

TEST_F(ProgramSpecTestGen1, CPU_DuplicateTensorParameterNameFails) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();

    auto binding_a = MakeMinimalTensorParameter("input_tensor");
    auto binding_b = MakeMinimalTensorParameter("input_tensor");  // duplicate!
    spec.tensor_parameters = {binding_a, binding_b};

    // Bind one of them to a kernel so the "every binding must be bound" check is satisfied;
    // the duplicate-name check fires first regardless.
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Duplicate TensorParameter name 'input_tensor'")));
}

TEST_F(ProgramSpecTestQuasar, CPU_EmptyKernelsFails) {
    ProgramSpec spec;
    spec.name = "empty_program";
    spec.work_units = std::vector<WorkUnitSpec>{};  // Empty work_units too

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("A ProgramSpec must have at least one KernelSpec")));
}

TEST_F(ProgramSpecTestQuasar, CPU_EmptyWorkUnitSpecsFails) {
    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel = MakeMinimalGen2DMKernel("kernel");
    spec.kernels = {kernel};
    spec.work_units = std::vector<WorkUnitSpec>{};  // Empty!

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'kernel' is not referenced by any WorkUnitSpec")));
}

TEST_F(ProgramSpecTestQuasar, CPU_OverlappingWorkUnitSpecsFails) {
    // Two work_units cannot target overlapping nodes
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel1 = MakeMinimalGen2DMKernel("kernel1");
    auto kernel2 = MakeMinimalGen2DMKernel("kernel2");
    spec.kernels = {kernel1, kernel2};

    // Both work_units target the same node
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("work_unit1", node, {"kernel1"}), MakeMinimalWorkUnit("work_unit2", node, {"kernel2"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("overlap in target nodes")));
}

TEST_F(ProgramSpecTestQuasar, CPU_MultipleWorkUnitsOnDifferentNodesSucceeds) {
    // Multiple work_units on non-overlapping nodes
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};
    NodeRangeSet all_nodes(std::set<NodeRange>{NodeRange{node0, node0}, NodeRange{node1, node1}});

    ProgramSpec spec;
    spec.name = "multi_work_unit_program";

    // Kernels span both nodes
    auto kernel = MakeMinimalGen2DMKernel("kernel");
    spec.kernels = {kernel};

    // Two work_units, each on a different node
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("work_unit0", node0, {"kernel"}), MakeMinimalWorkUnit("work_unit1", node1, {"kernel"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

// Cross-node DFBs are part of the API surface but not yet supported by the runtime.
TEST_F(ProgramSpecTestQuasar, CPU_CrossNodeDFBNotYetSupportedAtRuntime) {
    NodeCoord producer_node{0, 0};
    NodeCoord consumer_node{1, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer = MakeMinimalGen2DMKernel("consumer");

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer, consumer};
    spec.cross_node_dataflow_buffers = {CrossNodeDataflowBufferSpec{
        .dfb_spec = MakeMinimalDFB("dfb"),
        .producer_consumer_map = {{producer_node, consumer_node}},
    }};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("producer_work_unit", producer_node, {"producer"}),
        MakeMinimalWorkUnit("consumer_work_unit", consumer_node, {"consumer"}),
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("not yet supported")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
