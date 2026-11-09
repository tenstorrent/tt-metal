// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ProgramSpec structural invariants on DFBAdvancedOptions::alias_with (program_spec.hpp): alias groups
// are symmetric and agree on total size, node set, borrowed_from, arena parameter, and byte offset.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <optional>
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

ProgramSpec MakeArenaAliasSpecForTest() {
    auto first = MakeMinimalDFB("first", 512, 8);
    auto second = MakeMinimalDFB("second", 512, 8);
    first.advanced_options.alias_with = {DFBSpecName{"second"}};
    second.advanced_options.alias_with = {DFBSpecName{"first"}};
    ProgramSpec spec = MakeAliasProgramSpec(NodeCoord{0, 0}, first, second);
    spec.advanced_options.sram_arena_parameters = {
        {.unique_id = SRAMArenaParamName{"first_arena"}, .properties = LockstepSRAMArenaProperties{8192}},
        {.unique_id = SRAMArenaParamName{"second_arena"}, .properties = LockstepSRAMArenaProperties{8192}}};
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

TEST_F(ProgramSpecTestQuasar, CPU_SRAMArenaAliasedDFBsUseSameArena) {
    // Aliased DFBs can allocate from the same arena parameter.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    ProgramSpec spec = MakeArenaAliasSpecForTest();
    spec.advanced_options.sram_arena_parameters.resize(1);
    spec.dataflow_buffers[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"first_arena"}};
    spec.dataflow_buffers[1].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"first_arena"}};
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_SRAMArenaAliasedDFBsUseSameNonzeroOffset) {
    // Aliased DFBs can allocate at the same nonzero offset in an arena.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    ProgramSpec spec = MakeArenaAliasSpecForTest();
    spec.advanced_options.sram_arena_parameters.resize(1);
    spec.dataflow_buffers[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"first_arena"}, .offset = 4096};
    spec.dataflow_buffers[1].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"first_arena"}, .offset = 4096};
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_SRAMArenaAliasedDFBsRejectDifferentOffsets) {
    // Aliased DFBs must use the same byte offset in an arena.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    ProgramSpec spec = MakeArenaAliasSpecForTest();
    spec.dataflow_buffers[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"first_arena"}, .offset = 0};
    spec.dataflow_buffers[1].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"first_arena"}, .offset = 4096};
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(fixture_device, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("different allocation_override offsets")));
}

TEST_F(ProgramSpecTestQuasar, CPU_SRAMArenaAliasedDFBsRejectDifferentArenas) {
    // Aliased DFBs must allocate from the same arena parameter.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    ProgramSpec spec = MakeArenaAliasSpecForTest();
    spec.dataflow_buffers[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"first_arena"}};
    spec.dataflow_buffers[1].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"second_arena"}};
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(fixture_device, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("inconsistent allocation_override arena parameters")));
}

TEST_F(ProgramSpecTestQuasar, CPU_SRAMArenaAliasedDFBsRejectDefaultAllocation) {
    // An alias group cannot mix default allocation with an arena allocation override.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    ProgramSpec spec = MakeArenaAliasSpecForTest();
    spec.dataflow_buffers[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"first_arena"}};
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(fixture_device, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("inconsistent allocation_override arena parameters")));
}

TEST_F(ProgramSpecTestQuasar, CPU_SRAMArenaAliasedDFBsRejectClearedOverrideOnOneMember) {
    // An alias group is rejected when one member keeps an arena override and another member's override was cleared.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    ProgramSpec spec = MakeArenaAliasSpecForTest();
    spec.advanced_options.sram_arena_parameters.resize(1);
    const SRAMArenaAllocation allocation{.arena = SRAMArenaParamName{"first_arena"}};
    spec.dataflow_buffers[0].advanced_options.allocation_override = allocation;
    spec.dataflow_buffers[1].advanced_options.allocation_override = allocation;
    spec.dataflow_buffers[1].advanced_options.allocation_override = std::nullopt;
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(fixture_device, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("inconsistent allocation_override arena parameters")));
}

TEST_F(ProgramSpecTestQuasar, CPU_AliasDFBsWithClearedArenaOverridesUseDefaultAllocation) {
    // Aliased DFBs whose different arena overrides were cleared build as a default-allocated alias group.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    ProgramSpec spec = MakeArenaAliasSpecForTest();
    spec.advanced_options.sram_arena_parameters.clear();
    spec.dataflow_buffers[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"first_arena"}, .offset = 0};
    spec.dataflow_buffers[1].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"second_arena"}, .offset = 4096};
    spec.dataflow_buffers[0].advanced_options.allocation_override = std::nullopt;
    spec.dataflow_buffers[1].advanced_options.allocation_override = std::nullopt;
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device, spec));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
