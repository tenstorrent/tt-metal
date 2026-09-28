// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ProgramSpec structural invariants on placement per WorkUnitSpec (program_spec.hpp): each node hosting
// a DFB has exactly one producer and one consumer instance, and at most one kernel per node binds a
// given scratchpad.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "metal2_host_api/test_helpers.hpp"
#include "metal2_host_api/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen1ComputeKernel;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalGen2ComputeKernel;
using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestGen1;
using test_helpers::ProgramSpecTestQuasar;

TEST_F(ProgramSpecTestQuasar, CPU_DFBWithOnlyProducerFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel = MakeMinimalGen2DMKernel("kernel");
    auto dfb = MakeMinimalDFB("dfb");

    // Only bind as producer, no consumer
    kernel.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "accessor"));

    spec.kernels = {kernel};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("DFB 'dfb' has no consumer")));
}

TEST_F(ProgramSpecTestQuasar, CPU_DFBWithOnlyConsumerFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel = MakeMinimalGen2DMKernel("kernel");
    auto dfb = MakeMinimalDFB("dfb");

    // Only bind as consumer, no producer
    kernel.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "accessor"));

    spec.kernels = {kernel};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("DFB 'dfb' has no producer")));
}

TEST_F(ProgramSpecTestQuasar, CPU_DFBWithMultipleProducersInSameWorkUnitFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer1 = MakeMinimalGen2DMKernel("producer1");
    auto producer2 = MakeMinimalGen2DMKernel("producer2");
    auto consumer = MakeMinimalGen2ComputeKernel("consumer");

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    // Two PRODUCER bindings on the same DFB, both KernelSpecs in the same WorkUnitSpec (so both
    // land on the same node). A local DFB allows only one producer instance per node, so this fails.
    producer1.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    producer2.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer1, producer2, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units =
        std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer1", "producer2", "consumer"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("2 producer instance(s)")));
}

TEST_F(ProgramSpecTestQuasar, CPU_DFBWithMultipleConsumersInSameWorkUnitFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    // Both consumers DM (same kind) so the per-role kind-uniformity check passes and the
    // WU-disjointness check is what fires.
    auto consumer1 = MakeMinimalGen2DMKernel("consumer1");
    auto consumer2 = MakeMinimalGen2DMKernel("consumer2");

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer1.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));
    consumer2.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer, consumer1, consumer2};
    spec.dataflow_buffers = {dfb};
    spec.work_units =
        std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer1", "consumer2"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("2 consumer instance(s)")));
}

TEST_F(ProgramSpecTestQuasar, CPU_DFBWithMultipleConsumersInDifferentWorkUnitsSucceeds) {
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer1 = MakeMinimalGen2ComputeKernel("consumer1");
    auto consumer2 = MakeMinimalGen2ComputeKernel("consumer2");

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer1.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));
    consumer2.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    // producer covers both WUs (placed in both); each consumer covers one WU.
    spec.kernels = {producer, consumer1, consumer2};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("wu_g1", node0, {"producer", "consumer1"}),
        MakeMinimalWorkUnit("wu_g2", node1, {"producer", "consumer2"}),
    };

    EXPECT_NO_THROW({ MakeProgramFromSpec(*mesh_device_, spec); });
}

TEST_F(ProgramSpecTestQuasar, CPU_DFBWithMultipleProducersInDifferentWorkUnitsSucceeds) {
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer1 = MakeMinimalGen2DMKernel("producer1");
    auto producer2 = MakeMinimalGen2DMKernel("producer2");
    auto consumer = MakeMinimalGen2ComputeKernel("consumer");

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    producer1.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    producer2.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer1, producer2, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("wu_g1", node0, {"producer1", "consumer"}),
        MakeMinimalWorkUnit("wu_g2", node1, {"producer2", "consumer"}),
    };

    EXPECT_NO_THROW({ MakeProgramFromSpec(*mesh_device_, spec); });
}

TEST_F(ProgramSpecTestQuasar, CPU_DFBProducerConsumerCoverageMismatchFails) {
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer = MakeMinimalGen2ComputeKernel("consumer");

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    // Producer covers node0; consumer covers node1. Every node ends up with only one role —
    // the per-node census rejects it (each node hosting the DFB needs both a producer and consumer).
    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("wu_p", node0, {"producer"}),
        MakeMinimalWorkUnit("wu_c", node1, {"consumer"}),
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("is malformed at node")));
}

TEST_F(ProgramSpecTestQuasar, CPU_LocalDFBConsumerOnNodeWithoutProducerFails) {
    // A local DFB requires every node it lives on to host both a producer and a consumer. Here the
    // consumer covers an extra node where no producer runs, so the per-node census rejects it.
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer = MakeMinimalGen2DMKernel("consumer");
    auto dfb = MakeMinimalDFB("dfb");

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    // Producer covers node0 only; consumer covers node0 and node1 — node1 has a consumer but no producer.
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("work_unit_0", node0, {"producer", "consumer"}),
        MakeMinimalWorkUnit("work_unit_1", node1, {"consumer"}),
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("This node has a consumer but no producer")));
}

// The canonical 2D-matmul placement: one compute consumer spanning the whole grid, with a
// separate specialized DM producer per node group. The producer role has several KernelSpecs (one
// per group's WorkUnitSpec); the single consumer joins every group's WorkUnitSpec. Each node ends
// up with exactly one producer and one consumer instance, so the per-node census accepts it.
TEST_F(ProgramSpecTestQuasar, CPU_LocalDFBAllGridConsumerWithPerGroupProducersSucceeds) {
    ProgramSpec spec;
    spec.name = "test_program";

    auto consumer = MakeMinimalGen2ComputeKernel("compute");
    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    std::vector<KernelSpec> kernels;
    std::vector<WorkUnitSpec> work_units;
    for (uint32_t i = 0; i < 4; ++i) {
        const std::string dm_name = "dm" + std::to_string(i);
        auto dm = MakeMinimalGen2DMKernel(dm_name);
        dm.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
        kernels.push_back(dm);
        // One node per group: the per-group DM lives there, and the all-grid compute joins it.
        work_units.push_back(MakeMinimalWorkUnit("wu" + std::to_string(i), NodeCoord{i, 0}, {dm_name, "compute"}));
    }
    kernels.push_back(consumer);

    spec.kernels = std::move(kernels);
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::move(work_units);

    EXPECT_NO_THROW({ MakeProgramFromSpec(*mesh_device_, spec); });
}

TEST_F(ProgramSpecTestGen1, CPU_DFBMultipleProducersOnSameNodeFailsWithoutFlag) {
    // Baseline: without allow_instance_multi_binding, two producer instances on one node are rejected
    // by the per-node census even on Gen1. This is the counterpart the escape-hatch test unlocks.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "multi_producer";

    auto producer1 = MakeMinimalGen1DMKernel("producer1", DataMovementProcessor::RISCV_0);
    auto producer2 = MakeMinimalGen1DMKernel("producer2", DataMovementProcessor::RISCV_1);
    auto consumer = MakeMinimalGen1ComputeKernel("consumer");

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    producer1.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    producer2.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer1, producer2, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units =
        std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer1", "producer2", "consumer"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("2 producer instance(s)")));
}

TEST_F(ProgramSpecTestGen1, CPU_DFBMultipleProducersOnSameNodeSucceedsWithFlag) {
    // The allow_instance_multi_binding escape hatch: on Gen1 a DFB lowers to a plain circular buffer,
    // so one node may host more than one producer instance — here a RISCV_0 and a RISCV_1 DM kernel
    // both feeding one DFB, drained by a compute consumer. Identical to the FailsWithoutFlag case
    // except for the flag, which is what makes the per-node census accept the second producer.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "multi_producer";

    auto producer1 = MakeMinimalGen1DMKernel("producer1", DataMovementProcessor::RISCV_0);
    auto producer2 = MakeMinimalGen1DMKernel("producer2", DataMovementProcessor::RISCV_1);
    auto consumer = MakeMinimalGen1ComputeKernel("consumer");

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    dfb.advanced_options.allow_instance_multi_binding = true;

    producer1.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    producer2.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer1, producer2, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units =
        std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer1", "producer2", "consumer"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestGen1, CPU_DFBMultipleConsumersOnSameNodeSucceedsWithFlag) {
    // Mirror of the multi-producer escape-hatch case: a compute producer feeding two DM consumers
    // (RISCV_0 and RISCV_1) on one node. (Two DM consumers exhaust both DM processors, so the
    // producer must be the compute engine.) Unlocked by the flag.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "multi_consumer";

    auto producer = MakeMinimalGen1ComputeKernel("producer");
    auto consumer1 = MakeMinimalGen1DMKernel("consumer1", DataMovementProcessor::RISCV_0);
    auto consumer2 = MakeMinimalGen1DMKernel("consumer2", DataMovementProcessor::RISCV_1);

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    dfb.advanced_options.allow_instance_multi_binding = true;

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer1.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));
    consumer2.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer, consumer1, consumer2};
    spec.dataflow_buffers = {dfb};
    spec.work_units =
        std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer1", "consumer2"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadBoundByTwoKernelsSameNodeFails) {
    // One ScratchpadSpec bound by two kernels that share a node. A scratchpad is private node-local
    // L1; binding it from two kernels on the SAME node would be true sharing, which is not yet
    // supported (the disjoint-node case IS allowed — see the next test). MakeMinimalValidProgramSpec
    // places both kernels on node {0,0}, so this is the same-node collision case.
    ProgramSpec spec = MakeMinimalValidProgramSpec();

    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch_0"}, .size_per_node = 1024}};
    // kernels[0] (DM) and kernels[1] (compute) both bind it, and both run on node {0,0}.
    spec.kernels[0].scratchpad_bindings = {KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"scratch_0"}, .accessor_name = "s_dm"}};
    spec.kernels[1].scratchpad_bindings = {KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"scratch_0"}, .accessor_name = "s_compute"}};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("kernel instances on node")));
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadBoundByTwoKernelsDisjointNodesSucceeds) {
    // Complement of the same-node case above: one ScratchpadSpec bound by two kernels on DISJOINT
    // nodes is legal. Each node hosts exactly one binding kernel instance, so the per-node scratchpad
    // stays private to that kernel (allocation + CRTA delivery are per-binding-kernel, so the two
    // bindings never interact). This is the matmul-grid-style fan: one kernel source specialized into
    // multiple KernelSpecs on disjoint node ranges, all binding the same scratchpad resource.
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};

    ProgramSpec spec;
    spec.name = "scratchpad_shared_disjoint";

    auto kernel_a = MakeMinimalGen2DMKernel("kernel_a");
    kernel_a.scratchpad_bindings.push_back(KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"scratch_shared"}, .accessor_name = "scratch"});
    auto kernel_b = MakeMinimalGen2DMKernel("kernel_b");
    kernel_b.scratchpad_bindings.push_back(KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"scratch_shared"}, .accessor_name = "scratch"});

    spec.kernels = {kernel_a, kernel_b};
    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch_shared"}, .size_per_node = 1024}};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("wu_a", node0, {"kernel_a"}),
        MakeMinimalWorkUnit("wu_b", node1, {"kernel_b"}),
    };

    EXPECT_NO_THROW({ MakeProgramFromSpec(*mesh_device_, spec); });
}

}  // namespace
}  // namespace tt::tt_metal::experimental
