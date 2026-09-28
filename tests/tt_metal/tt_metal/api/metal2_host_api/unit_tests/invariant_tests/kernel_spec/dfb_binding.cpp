// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Local invariants of KernelSpec::DFBBinding and KernelSpec::dfb_bindings (kernel_spec.hpp): accessor
// names, at most one PRODUCER and one CONSUMER binding per DFB, and self-loop rules.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <stdexcept>
#include <string>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "metal2_host_api/test_helpers.hpp"
#include "metal2_host_api/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalGen2ComputeKernel;
using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestGen1;
using test_helpers::ProgramSpecTestQuasar;

TEST_F(ProgramSpecTestQuasar, CPU_InvalidLocalAccessorNameFails) {
    NodeCoord node{0, 0};

    const std::vector<std::string> invalid_names = {
        "",               // empty
        "has-dash",       // hyphen
        "has space",      // whitespace
        "1starts_digit",  // leading digit
        "has.dot",        // punctuation
        "class",          // C++ keyword
        "namespace",      // C++ keyword
        "int",            // C++ keyword
        "_Foo",           // reserved: underscore + uppercase
        "__foo",          // reserved: leading double underscore
        "foo__bar",       // reserved: embedded double underscore
    };

    for (const auto& bad_name : invalid_names) {
        ProgramSpec spec;
        spec.name = "test_program";

        auto kernel = MakeMinimalGen2DMKernel("kernel");
        auto dfb = MakeMinimalDFB("dfb");

        kernel.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, bad_name));

        spec.kernels = {kernel};
        spec.dataflow_buffers = {dfb};
        spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

        EXPECT_THAT(
            [&] { MakeProgramFromSpec(*mesh_device_, spec); },
            ::testing::ThrowsMessage<std::runtime_error>(
                ::testing::HasSubstr("DFB accessor_name '" + bad_name + "' must be a valid C++ identifier")))
            << "Expected rejection for name: '" << bad_name << "'";
    }

    // A valid-but-too-long identifier cannot be passed to the kernel-side by-name binding lookup,
    // so it must fail at Program construction rather than as a kernel JIT static_assert.
    const std::string too_long(MAX_ACCESSOR_NAME_LENGTH + 1, 'a');
    ProgramSpec spec;
    spec.name = "test_program";
    auto kernel = MakeMinimalGen2DMKernel("kernel");
    auto dfb = MakeMinimalDFB("dfb");
    kernel.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, too_long));
    spec.kernels = {kernel};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("must be at most " + std::to_string(MAX_ACCESSOR_NAME_LENGTH) + " characters")));
}

TEST_F(ProgramSpecTestQuasar, CPU_SharedLocalAccessorNameForDifferentDFBsFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto kernel = MakeMinimalGen2DMKernel("kernel");
    auto dfb0 = MakeMinimalDFB("dfb_0");
    auto dfb1 = MakeMinimalDFB("dfb_1");

    // Bind two *different* DFBs with the same accessor_name — illegal
    // (self-loop sharing requires the same DFB on both bindings).
    kernel.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb_0"}, "same_accessor"));
    kernel.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb_1"}, "same_accessor"));

    spec.kernels = {kernel};
    spec.dataflow_buffers = {dfb0, dfb1};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Kernel 'kernel' uses accessor_name 'same_accessor' for two different DFBs")));
}

TEST_F(ProgramSpecTestQuasar, CPU_DuplicateProducerBindingForSameLocalAccessorNameFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer_kernel = MakeMinimalGen2DMKernel("producer");
    auto consumer_kernel = MakeMinimalGen2DMKernel("consumer");
    auto dfb = MakeMinimalDFB("dfb");

    // Two PRODUCER bindings on the same kernel sharing a accessor_name —
    // illegal: the self-loop relaxation requires opposite endpoint types.
    producer_kernel.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "shared"));
    producer_kernel.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "shared"));
    consumer_kernel.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {producer_kernel, consumer_kernel};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("duplicate PRODUCER binding for accessor_name 'shared'")));
}

TEST_F(ProgramSpecTestQuasar, CPU_DFBBoundTwiceInSameRoleUnderDifferentNamesFails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    // The wrong port of the legacy "one buffer, two names" CB-alias idiom: one kernel binds the
    // same DFB twice in the SAME role (here two CONSUMER bindings) under different accessor names,
    // yielding two accessors / DataflowBuffer objects for one FIFO. Forbidden — the right port is a
    // kernel-side handle alias over a single binding. (A producer+consumer self-loop is a different,
    // legitimate multi-binding and stays legal — see SelfLoopWithSharedLocalAccessorNameSucceeds.)
    auto producer = MakeMinimalGen2DMKernel("producer");
    auto consumer = MakeMinimalGen2DMKernel("consumer");
    auto dfb = MakeMinimalDFB("dfb");

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in_a"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in_b"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"producer", "consumer"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("has two CONSUMER bindings to DFB 'dfb' under different accessor names")));
}

TEST_F(ProgramSpecTestQuasar, CPU_SelfLoopWithSharedLocalAccessorNameSucceeds) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    // A kernel that both produces and consumes the same DFB may share a single
    // accessor_name across the PRODUCER and CONSUMER bindings. Uses a COMPUTE kernel: a compute
    // self-loop is the only legal self-loop (it lowers to the intra-Tensix packer->unpacker flow),
    // so it exercises the accessor-name relaxation through a path that survives validation. (A DM
    // self-loop is rejected — see DMKernelSelfLoopFails.)
    auto kernel = MakeMinimalGen2ComputeKernel("kernel");
    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    kernel.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "acc"));
    kernel.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "acc"));

    spec.kernels = {kernel};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_NO_THROW({ MakeProgramFromSpec(*mesh_device_, spec); });
}

TEST_F(ProgramSpecTestQuasar, CPU_DFBSelfLoopOnComputeKernelSucceeds) {
    // A compute kernel that self-loops a DFB (binds it as both producer and consumer) is legal: it
    // lowers to the intra-Tensix packer->unpacker flow (TensixScope::INTRA), which the Metal 2.0
    // layer applies automatically. There is no user-facing self-loop scope option.
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "compute_self_loop";

    auto compute = MakeMinimalGen2ComputeKernel("compute");

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;
    // INTRA-tensix self-loop: no DM endpoint, so the spec-to-impl translation produces
    // enable_{producer,consumer}_implicit_sync=false at the lower DFB layer automatically.

    compute.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    compute.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));

    spec.kernels = {compute};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"compute"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_DMKernelSelfLoopOnGen2Fails) {
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    // On Gen2, a data-movement kernel may NOT self-loop a DFB (bind it as both PRODUCER and CONSUMER).
    // The DFB's tile-counter credit machinery synchronizes a producer and a consumer on DISTINCT RISCs
    // via per-side masks, and a single DM kernel's producer and consumer masks are identical — the DFB
    // backend would reject it with an opaque "producer_risc_mask and consumer_risc_mask must not
    // overlap". Caught up front at validation with an actionable message instead. The legal Gen2
    // alternatives are a private L1 scratch buffer, a LocalTensorAccessor tensor view, or a two-kernel
    // cross-bind. (On Gen1 a DM self-loop IS legal — a DFB lowers to a plain circular buffer there; see
    // DMKernelSelfLoopOnGen1Succeeds. Compute self-loops stay legal on both gens — see
    // DFBSelfLoopOnComputeKernelSucceeds.)
    auto kernel = MakeMinimalGen2DMKernel("kernel");
    auto dfb = MakeMinimalDFB("dfb");
    kernel.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "p"));
    kernel.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "c"));

    spec.kernels = {kernel};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::AllOf(
            ::testing::HasSubstr("self-looped by data-movement kernel 'kernel'"),
            ::testing::HasSubstr("not supported for data-movement kernels on Gen2 architectures"))));
}

TEST_F(ProgramSpecTestGen1, CPU_DMKernelSelfLoopOnGen1Succeeds) {
    // On Gen1 (WH/BH) a DFB lowers to a plain circular buffer, so a single DM kernel may bind it as
    // both PRODUCER and CONSUMER (self-loop) — the classic scratch pattern of one DM engine filling
    // and draining an L1 FIFO. There is no tile-counter credit machinery requiring disjoint
    // producer/consumer masks, so the spec validator accepts it. (On Gen2 the same spec is rejected —
    // see DMKernelSelfLoopOnGen2Fails.)
    NodeCoord node{0, 0};

    ProgramSpec spec;
    spec.name = "dm_self_loop";

    auto kernel = MakeMinimalGen1DMKernel("kernel");
    auto dfb = MakeMinimalDFB("dfb");
    kernel.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "p"));
    kernel.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "c"));

    spec.kernels = {kernel};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", node, {"kernel"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
