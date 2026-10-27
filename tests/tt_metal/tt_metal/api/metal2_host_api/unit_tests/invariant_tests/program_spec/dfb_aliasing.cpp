// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ProgramSpec structural invariants on DFBAdvancedOptions::alias_with (program_spec.hpp): alias groups
// are symmetric and agree on total size, node set and borrowed_from.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <stdexcept>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindTensorParameterToKernel;
using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalTensorParameter;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestQuasar;

// Helper: build a minimal 1-producer / 1-consumer ProgramSpec where both DFBs are
// bound to the same producer/consumer kernels in a single WorkUnit on a single node.
namespace {
ProgramSpec MakeAliasProgramSpec(
    const NodeCoord& node, const DataflowBufferSpec& dfb_a, const DataflowBufferSpec& dfb_b) {
    ProgramSpec spec;

    KernelSpec producer = MakeMinimalGen2DMKernel("producer_kernel");
    KernelSpec consumer = MakeMinimalGen2DMKernel("consumer_kernel");

    producer.dfb_bindings.push_back(ProducerOf(dfb_a.unique_id, "out_a"));
    consumer.dfb_bindings.push_back(ConsumerOf(dfb_a.unique_id, "in_a"));

    producer.dfb_bindings.push_back(ProducerOf(dfb_b.unique_id, "out_b"));
    consumer.dfb_bindings.push_back(ConsumerOf(dfb_b.unique_id, "in_b"));

    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb_a, dfb_b};
    spec.work_units = {MakeMinimalWorkUnit("wu", node, {"producer_kernel", "consumer_kernel"})};
    return spec;
}
}  // anonymous namespace

TEST_F(ProgramSpecTestQuasar, CPU_AliasDFBFailsOnMismatchedTotalSize) {
    // DFB_A: 512 * 8 = 4096 bytes, DFB_B: 256 * 8 = 2048 bytes — different totals → TT_FATAL
    auto dfb_a = MakeMinimalDFB("dfb_a", /*entry_size=*/512, /*num_entries=*/8);
    auto dfb_b = MakeMinimalDFB("dfb_b", /*entry_size=*/256, /*num_entries=*/8);
    dfb_a.advanced_options = DFBAdvancedOptions{.alias_with = {DFBSpecName{"dfb_b"}}};
    dfb_b.advanced_options = DFBAdvancedOptions{.alias_with = {DFBSpecName{"dfb_a"}}};

    const NodeCoord node{0, 0};
    auto spec = MakeAliasProgramSpec(node, dfb_a, dfb_b);

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("different total sizes")));
}

TEST_F(ProgramSpecTestQuasar, CPU_AliasDFBFailsOnAsymmetricDeclaration) {
    // DFB_A lists DFB_B but DFB_B does not list DFB_A — clique violation → TT_FATAL
    auto dfb_a = MakeMinimalDFB("dfb_a", /*entry_size=*/512, /*num_entries=*/8);
    auto dfb_b = MakeMinimalDFB("dfb_b", /*entry_size=*/256, /*num_entries=*/16);
    dfb_a.advanced_options = DFBAdvancedOptions{.alias_with = {DFBSpecName{"dfb_b"}}};
    // dfb_b.alias_with intentionally left empty

    const NodeCoord node{0, 0};
    auto spec = MakeAliasProgramSpec(node, dfb_a, dfb_b);

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("do not declare the same alias group")));
}

TEST_F(ProgramSpecTestQuasar, CPU_AliasDFBMatmulStyleSucceeds) {
    // This is the nasty case from the matmul op....
    // Two DFBs share L1 but are bound to different kernels.
    //  - DFB_A is bound to {producer_kernel, consumer_kernel};
    //  - DFB_B is bound to {producer_kernel, other_kernel}.
    //
    // This looks unspeakably evil and I'd like to forbid it. But, it does work.
    // All kernels run on the same node set, so they all have the same L1.
    // And (presumably) the DFB is used in a temporally disjoint way.
    // So, nothing stops them from re-using the DFB memory.

    const NodeCoord node{0, 0};

    auto dfb_a = MakeMinimalDFB("dfb_a", /*entry_size=*/512, /*num_entries=*/8);
    auto dfb_b = MakeMinimalDFB("dfb_b", /*entry_size=*/512, /*num_entries=*/8);
    dfb_a.advanced_options = DFBAdvancedOptions{.alias_with = {DFBSpecName{"dfb_b"}}};
    dfb_b.advanced_options = DFBAdvancedOptions{.alias_with = {DFBSpecName{"dfb_a"}}};

    KernelSpec producer = MakeMinimalGen2DMKernel("producer_kernel");
    KernelSpec consumer = MakeMinimalGen2DMKernel("consumer_kernel");
    KernelSpec other = MakeMinimalGen2DMKernel("other_kernel");

    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb_a"}, "out_a"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb_a"}, "in_a"));
    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb_b"}, "out_b"));
    other.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb_b"}, "in_b"));

    ProgramSpec spec;
    spec.kernels = {producer, consumer, other};
    spec.dataflow_buffers = {dfb_a, dfb_b};
    spec.work_units = {MakeMinimalWorkUnit("wu", node, {"producer_kernel", "consumer_kernel", "other_kernel"})};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_AliasDFBFailsOnDifferentNodeCoverage) {
    // Two DFBs aliased, but bound to kernels running on disjoint nodes. The shared L1
    // region only makes sense if both members cover the same cores; otherwise the
    // secondary's address propagation would alias into L1 the primary never reserved.
    NodeCoord node_a{0, 0};
    NodeCoord node_b{1, 0};

    auto dfb_a = MakeMinimalDFB("dfb_a", /*entry_size=*/512, /*num_entries=*/8);
    auto dfb_b = MakeMinimalDFB("dfb_b", /*entry_size=*/512, /*num_entries=*/8);
    dfb_a.advanced_options = DFBAdvancedOptions{.alias_with = {DFBSpecName{"dfb_b"}}};
    dfb_b.advanced_options = DFBAdvancedOptions{.alias_with = {DFBSpecName{"dfb_a"}}};

    KernelSpec producer_a = MakeMinimalGen2DMKernel("producer_a");
    KernelSpec consumer_a = MakeMinimalGen2DMKernel("consumer_a");
    KernelSpec producer_b = MakeMinimalGen2DMKernel("producer_b");
    KernelSpec consumer_b = MakeMinimalGen2DMKernel("consumer_b");
    producer_a.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb_a"}, "out_a"));
    consumer_a.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb_a"}, "in_a"));
    producer_b.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb_b"}, "out_b"));
    consumer_b.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb_b"}, "in_b"));

    ProgramSpec spec;
    spec.kernels = {producer_a, consumer_a, producer_b, consumer_b};
    spec.dataflow_buffers = {dfb_a, dfb_b};
    spec.work_units = {
        MakeMinimalWorkUnit("wu_a", node_a, {"producer_a", "consumer_a"}),
        MakeMinimalWorkUnit("wu_b", node_b, {"producer_b", "consumer_b"}),
    };

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("cover different sets of nodes")));
}

TEST_F(ProgramSpecTestQuasar, CPU_AliasDFBFailsOnInconsistentBorrowedFrom) {
    // DFB_A borrows from a TensorParameter, DFB_B does not. Within an alias group, either
    // no member borrows or all members borrow from the same TensorParameter.
    const NodeCoord node{0, 0};

    auto dfb_a = MakeMinimalDFB("dfb_a", /*entry_size=*/16, /*num_entries=*/2);
    auto dfb_b = MakeMinimalDFB("dfb_b", /*entry_size=*/16, /*num_entries=*/2);
    dfb_a.advanced_options = DFBAdvancedOptions{.alias_with = {DFBSpecName{"dfb_b"}}};
    dfb_b.advanced_options = DFBAdvancedOptions{.alias_with = {DFBSpecName{"dfb_a"}}};
    dfb_a.borrowed_from = TensorParamName{"borrowed_tensor"};
    // dfb_b.borrowed_from intentionally left unset

    KernelSpec producer = MakeMinimalGen2DMKernel("producer_kernel");
    KernelSpec consumer = MakeMinimalGen2DMKernel("consumer_kernel");
    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb_a"}, "out_a"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb_a"}, "in_a"));
    producer.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb_b"}, "out_b"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb_b"}, "in_b"));

    auto tensor_param = MakeMinimalTensorParameter("borrowed_tensor", tt::tt_metal::BufferType::L1);
    BindTensorParameterToKernel(producer, "borrowed_tensor", "borrowed_t");

    ProgramSpec spec;
    spec.kernels = {producer, consumer};
    spec.dataflow_buffers = {dfb_a, dfb_b};
    spec.tensor_parameters = {tensor_param};
    spec.work_units = {MakeMinimalWorkUnit("wu", node, {"producer_kernel", "consumer_kernel"})};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("inconsistent borrowed_from")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
