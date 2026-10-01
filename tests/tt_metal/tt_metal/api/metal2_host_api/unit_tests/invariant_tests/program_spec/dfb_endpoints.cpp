// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ProgramSpec structural invariants on DataflowBufferSpec endpoints (program_spec.hpp): every bound DFB has
// a producer and a consumer; same-role bindings agree on access pattern, num_threads, kind and (Gen1)
// processor; self-loop sides match; Gen2 DM sides agree on implicit sync.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <stdexcept>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen1ComputeKernel;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalGen2ComputeKernel;
using test_helpers::MakeMinimalGen2DMKernel;
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

TEST_F(ProgramSpecTestQuasar, CPU_DFBMultiBindingAccessPatternMismatchFails) {
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
    consumer2.dfb_bindings.push_back(AllConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer, consumer1, consumer2};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("wu_g1", node0, {"producer", "consumer1"}),
        MakeMinimalWorkUnit("wu_g2", node1, {"producer", "consumer2"}),
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("DFB 'dfb' has multiple CONSUMER bindings with mismatched access_pattern")));
}

TEST_F(ProgramSpecTestQuasar, CPU_DFBMultiBindingNumThreadsMismatchFails) {
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer1 = MakeMinimalGen2ComputeKernel("consumer1", /*num_threads=*/1);
    auto consumer2 = MakeMinimalGen2ComputeKernel("consumer2", /*num_threads=*/2);

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer1.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));
    consumer2.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer, consumer1, consumer2};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("wu_g1", node0, {"producer", "consumer1"}),
        MakeMinimalWorkUnit("wu_g2", node1, {"producer", "consumer2"}),
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("DFB 'dfb' has multiple CONSUMER KernelSpecs with mismatched num_threads")));
}

TEST_F(ProgramSpecTestQuasar, CPU_DFBMultiBindingMixingComputeAndDMOnSameRoleFails) {
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    // Producer side mixes a DM and a compute kernel on disjoint zones. Each individually
    // would form a valid binding, but the DFB's hardware config carries a single producer
    // processor mask per role; the two kinds occupy disjoint mask bit ranges and cannot
    // share a mask. The validator must reject upfront.
    auto dm_producer = MakeMinimalGen2DMKernel("dm_producer");
    auto compute_producer = MakeMinimalGen2ComputeKernel("compute_producer");
    auto consumer = MakeMinimalGen2DMKernel("consumer");

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    dm_producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    compute_producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {dm_producer, compute_producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("wu_g1", node0, {"dm_producer", "consumer"}),
        MakeMinimalWorkUnit("wu_g2", node1, {"compute_producer", "consumer"}),
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("mixing compute and data-movement kinds")));
}

TEST_F(ProgramSpecTestGen1, CPU_DFBMixedKindProducersOnSameNodeFailsWithoutFlag) {
    // Baseline: a single role mixing a compute and a DM kernel is rejected by the per-role
    // kind-uniformity check (the DFB's hardware config nominally carries one processor mask per role).
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "mixed_kind";

    auto producer_compute = MakeMinimalGen1ComputeKernel("producer_compute");
    auto producer_dm = MakeMinimalGen1DMKernel("producer_dm", DataMovementProcessor::RISCV_0);
    auto consumer = MakeMinimalGen1DMKernel("consumer", DataMovementProcessor::RISCV_1);

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    producer_compute.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    producer_dm.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer_compute, producer_dm, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("work_unit", node, {"producer_compute", "producer_dm", "consumer"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("mixing compute and data-movement kinds")));
}

TEST_F(ProgramSpecTestGen1, CPU_DFBMixedKindProducersOnSameNodeSucceedsWithFlag) {
    // The escape hatch imposes no RISC-type restriction on Gen1: a compute kernel and a DM kernel may
    // both produce to one DFB on one node (it lowers to a plain shared circular buffer). Identical to
    // the FailsWithoutFlag case except for the flag, which is what drops the kind-uniformity check.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "mixed_kind";

    auto producer_compute = MakeMinimalGen1ComputeKernel("producer_compute");
    auto producer_dm = MakeMinimalGen1DMKernel("producer_dm", DataMovementProcessor::RISCV_0);
    auto consumer = MakeMinimalGen1DMKernel("consumer", DataMovementProcessor::RISCV_1);

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    dfb.advanced_options.allow_instance_multi_binding = true;

    producer_compute.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    producer_dm.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer_compute, producer_dm, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("work_unit", node, {"producer_compute", "producer_dm", "consumer"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

// On Gen1 (WH/BH) the per-kernel risc_mask is a deterministic function of the
// KernelSpec's config. Multi-binding requires all same-role KernelSpecs to
// share that mask; mismatched processor placement on the producer (or consumer)
// side is a user error and must be rejected with an actionable message.
TEST_F(ProgramSpecTestGen1, CPU_MultiBindingProducerMaskMismatchFails) {
    NodeCoord node0{0, 0};
    NodeCoord node1{0, 1};

    auto producer_g1 = MakeMinimalGen1DMKernel("producer_g1", DataMovementProcessor::RISCV_0);
    auto producer_g2 = MakeMinimalGen1DMKernel("producer_g2", DataMovementProcessor::RISCV_1);
    auto consumer = MakeMinimalGen1ComputeKernel("consumer");

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    producer_g1.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    producer_g2.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    ProgramSpec spec;
    spec.name = "multi_binding_mask_mismatch";
    spec.kernels = {producer_g1, producer_g2, consumer};
    spec.dataflow_buffers = {dfb};
    // consumer in both WUs (single-KernelSpec multi-WU membership) → consumer-side mask is fine.
    // producer_g1 in wu_g1 (RISCV_0); producer_g2 in wu_g2 (RISCV_1) → mismatched producer masks.
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("wu_g1", node0, {"producer_g1", "consumer"}),
        MakeMinimalWorkUnit("wu_g2", node1, {"producer_g2", "consumer"}),
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("DFB 'dfb' has multiple PRODUCER KernelSpecs ('producer_g1', 'producer_g2') with "
                                 "mismatched processor placement")));
}

TEST_F(ProgramSpecTestQuasar, CPU_DFBMultiBindingSelfLoopWithMatchingSidesSucceeds) {
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    // Two self-looping kernels on disjoint WUs. Each binds "dfb" as both producer and
    // consumer; producer set equals consumer set = {self_loop_1, self_loop_2}. At each node,
    // exactly one kernel runs and self-loops the DFB — the local invariant holds.
    auto self_loop_1 = MakeMinimalGen2ComputeKernel("self_loop_1");
    auto self_loop_2 = MakeMinimalGen2ComputeKernel("self_loop_2");

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    // INTRA-tensix self-loop DFBs have no DM endpoint; the spec-to-impl translation produces
    // enable_{producer,consumer}_implicit_sync=false automatically (no DM kernel to vote for it).

    self_loop_1.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "p"));
    self_loop_1.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "c"));
    self_loop_2.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "p"));
    self_loop_2.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "c"));

    spec.kernels = {self_loop_1, self_loop_2};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("wu_g1", node0, {"self_loop_1"}),
        MakeMinimalWorkUnit("wu_g2", node1, {"self_loop_2"}),
    };

    EXPECT_NO_THROW({ MakeProgramFromSpec(*mesh_device_, spec); });
}

TEST_F(ProgramSpecTestQuasar, CPU_DFBSelfLoopWithExtraProducerSideKernelFails) {
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    // self_loop_1 binds "dfb" as BOTH producer and consumer (self-loop). On the other WU,
    // an unrelated producer-only kernel is bound, while extra_consumer covers the consume side.
    // Producer set = {self_loop_1, extra_producer}; consumer set = {self_loop_1, extra_consumer}.
    // The sets are not equal — the self-loop multi-binding rule rejects this mix.
    //
    // All three kernels are COMPUTE so the self-loop participant (self_loop_1) is a legal compute
    // self-loop — a DM self-loop would be rejected earlier on Gen2 (see DMKernelSelfLoopOnGen2Fails),
    // masking the rule under test. With compute kernels the per-role kind-uniformity check passes and
    // the self-loop set-equality refinement check is reached.
    auto self_loop_1 = MakeMinimalGen2ComputeKernel("self_loop_1");
    auto extra_producer = MakeMinimalGen2ComputeKernel("extra_producer");
    auto extra_consumer = MakeMinimalGen2ComputeKernel("extra_consumer");

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    self_loop_1.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "p"));
    self_loop_1.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "c"));
    extra_producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    extra_consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {self_loop_1, extra_producer, extra_consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("wu_g1", node0, {"self_loop_1"}),
        MakeMinimalWorkUnit("wu_g2", node1, {"extra_producer", "extra_consumer"}),
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("DFB 'dfb' is self-looped (some kernel appears as both producer and consumer), but "
                                 "the set of producer KernelSpecs differs from the set of consumer KernelSpecs")));
}

// ----------------------------------------------------------------------------
// DFB implicit-sync opt-out (Gen2)
// ----------------------------------------------------------------------------
// Implicit sync is ON by default for any DFB side that has a DM endpoint. A DM kernel can
// opt out per-DFB (disable_dfb_implicit_sync_for) or for all the DFBs it binds at once
// (disable_dfb_implicit_sync_for_all). These tests pin the per-kernel "all" hammer.

TEST_F(ProgramSpecTestQuasar, CPU_DisableImplicitSyncForAllDisagreementAcrossProducersFails) {
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer1 = MakeMinimalGen2DMKernel("producer1");
    auto producer2 = MakeMinimalGen2DMKernel("producer2");
    auto consumer = MakeMinimalGen2ComputeKernel("consumer");

    // producer1 hammers implicit sync off; producer2 leaves it on. Both bind the same DFB on
    // the producer side, so the per-side opt-out disagrees and validation must reject.
    std::get<DataMovementHardwareConfig>(producer1.hw_config).config_2xx =
        DataMovementHardwareConfig::DataMovement2XXConfig{.disable_dfb_implicit_sync_for_all = true};

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

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("disagreeing implicit-sync opt-out state")));
}

TEST_F(ProgramSpecTestQuasar, CPU_DisableImplicitSyncForAllAgreesWithExplicitList) {
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer1 = MakeMinimalGen2DMKernel("producer1");
    auto producer2 = MakeMinimalGen2DMKernel("producer2");
    auto consumer = MakeMinimalGen2ComputeKernel("consumer");

    // producer1 opts out via the per-kernel hammer; producer2 opts the same DFB out by name.
    // Both express the same per-side decision (disable), so they agree and the side lowers off.
    std::get<DataMovementHardwareConfig>(producer1.hw_config).config_2xx =
        DataMovementHardwareConfig::DataMovement2XXConfig{.disable_dfb_implicit_sync_for_all = true};
    std::get<DataMovementHardwareConfig>(producer2.hw_config).config_2xx =
        DataMovementHardwareConfig::DataMovement2XXConfig{.disable_dfb_implicit_sync_for = {DFBSpecName{"dfb"}}};

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

    auto program = MakeProgramFromSpec(*mesh_device_, spec);
    const uint32_t dfb_id = program.impl().get_dfb_handle("dfb");
    EXPECT_FALSE(program.impl().get_dataflow_buffer(dfb_id)->config.enable_producer_implicit_sync);
}

}  // namespace
}  // namespace tt::tt_metal::experimental
