// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <type_traits>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/dfb_placement_test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::ProgramSpecTestGen1;
using test_helpers::ProgramSpecTestQuasar;

template <typename Arch>
class DFBPlacementOverrideTest : public test_helpers::MockMeshDeviceFixture<Arch::value> {
protected:
    distributed::MeshDevice& fixture_device() { return *this->mesh_device_; }
};

using Architectures = ::testing::Types<
    std::integral_constant<tt::ARCH, tt::ARCH::WORMHOLE_B0>,
    std::integral_constant<tt::ARCH, tt::ARCH::QUASAR>,
    std::integral_constant<tt::ARCH, tt::ARCH::BLACKHOLE>>;
TYPED_TEST_SUITE(DFBPlacementOverrideTest, Architectures);

TYPED_TEST(DFBPlacementOverrideTest, CPU_DFBPlacementSeparatesSlotsWithMatchingMetadata) {
    // Shared endpoint kernels use separate slots even when disjoint DFBs have matching metadata.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeDisjointDFBPlacementsForTest(TypeParam::value);
    spec.dataflow_buffers[1].tile_format_metadata = Tile{};
    std::get<ComputeHardwareConfig>(spec.kernels[1].hw_config).unpack_modes = {
        {DFBSpecName{"dfb_1"}, UnpackMode::UnpackToSrc}};
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto first = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    const auto second = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_1"));
    EXPECT_NE(first->device_slot, second->device_slot);
}

TYPED_TEST(DFBPlacementOverrideTest, CPU_DFBPlacementAcceptsDifferentFormatsForSeparateKernels) {
    // Separate kernels on disjoint nodes can use different formats on the same device slot.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeSeparateDFBEndpointPairs(TypeParam::value);
    spec.dataflow_buffers[1].data_format_metadata = tt::DataFormat::Float32;
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto first = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    const auto second = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_1"));
    EXPECT_EQ(first->device_slot, second->device_slot);
}

TYPED_TEST(DFBPlacementOverrideTest, CPU_DFBPlacementSeparatesSlotsForDifferentFormats) {
    // A kernel uses separate slots for disjoint DFBs with different data formats.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeDisjointDFBPlacementsForTest(TypeParam::value);
    spec.dataflow_buffers[1].data_format_metadata = tt::DataFormat::Float32;
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto first = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    const auto second = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_1"));
    EXPECT_NE(first->device_slot, second->device_slot);
}

TYPED_TEST(DFBPlacementOverrideTest, CPU_DFBPlacementSeparatesSlotsForDifferentTiles) {
    // A kernel uses separate slots for disjoint DFBs with different tile shapes.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeDisjointDFBPlacementsForTest(TypeParam::value);
    spec.dataflow_buffers[1].tile_format_metadata = Tile({16, 32});
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto first = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    const auto second = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_1"));
    EXPECT_NE(first->device_slot, second->device_slot);
}

TYPED_TEST(DFBPlacementOverrideTest, CPU_DFBPlacementSeparatesSlotsForDifferentUnpackModes) {
    // A kernel uses separate slots for explicit and default unpack modes on disjoint DFBs.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeDisjointDFBPlacementsForTest(TypeParam::value);
    std::get<ComputeHardwareConfig>(spec.kernels[1].hw_config).enable_32_bit_dest = true;
    std::get<ComputeHardwareConfig>(spec.kernels[1].hw_config).unpack_modes = {
        {DFBSpecName{"dfb_0"}, UnpackMode::UnpackToDest}};
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto first = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    const auto second = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_1"));
    EXPECT_NE(first->device_slot, second->device_slot);
}

TEST_F(ProgramSpecTestQuasar, CPU_DFBPlacementSeparatesSlotsForUnboundMetadata) {
    // Overlapping endpoint placements use separate slots even without a shared DFB binding.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeSeparateDFBEndpointPairs(tt::ARCH::QUASAR);
    spec.work_units[1].kernels.push_back(KernelSpecName{"dm_kernel"});
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}));
    spec.dataflow_buffers[1].data_format_metadata = tt::DataFormat::Float32;
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto first = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    const auto second = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_1"));
    EXPECT_NE(first->device_slot, second->device_slot);
}

TEST_F(ProgramSpecTestGen1, CPU_DFBPlacementPreservesDifferentUnpackModes) {
    // A compute kernel retains the explicit and default unpack modes in distinct DFB slots.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeDisjointDFBPlacementsForTest(tt::ARCH::WORMHOLE_B0);
    std::get<ComputeHardwareConfig>(spec.kernels[1].hw_config).enable_32_bit_dest = true;
    std::get<ComputeHardwareConfig>(spec.kernels[1].hw_config).unpack_modes = {
        {DFBSpecName{"dfb_0"}, UnpackMode::UnpackToDest}};
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto kernel = program.impl().get_kernel_by_spec_name("compute_kernel");
    const auto config = std::get<ComputeConfig>(kernel->config());
    const auto first = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    const auto second = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_1"));
    EXPECT_EQ(config.unpack_to_dest_mode[first->device_slot], UnpackToDestMode::UnpackToDestFp32);
    EXPECT_EQ(config.unpack_to_dest_mode[second->device_slot], UnpackToDestMode::Default);
}

TYPED_TEST(DFBPlacementOverrideTest, CPU_DFBPlacementSerializesPartiallyOverlappingOverrides) {
    // A program produces device configuration for partially overlapping DFB storage placements.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeDFBPlacementSpec(TypeParam::value, 2);
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {2, 0});
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}, {1, 0}));
    spec.dataflow_buffers[1].advanced_options.override_placement = CoreRangeSet(CoreRange({1, 0}, {2, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    test_helpers::FinalizeMockDFBConfig(program);
    const auto& config = program.impl().get_program_config(test_helpers::TensixIndexForTest(program));
    std::vector<uint8_t> payload(config.dfb_size);
    EXPECT_GT(
        dfb::detail::serialize_dfb_config_for_core({2, 0}, program.impl().dataflow_buffers_on_core({2, 0}), payload),
        0u);
}

TEST_F(ProgramSpecTestQuasar, CPU_DFBPlacementAliasesAcceptSplitRanges) {
    // Aliases can describe the same nodes with merged and split ranges.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeDFBPlacementSpec(tt::ARCH::QUASAR, 2);
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    spec.dataflow_buffers[0].advanced_options.alias_with = {DFBSpecName{"dfb_1"}};
    spec.dataflow_buffers[1].advanced_options.alias_with = {DFBSpecName{"dfb_0"}};
    spec.dataflow_buffers[1].advanced_options.override_placement =
        CoreRangeSet(std::vector<CoreRange>{CoreRange({0, 0}), CoreRange({1, 0})});
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_DFBPlacementRejectsNodesWithoutEndpoints) {
    // A DFB cannot reserve a node with neither a producer nor a consumer.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeDFBPlacementSpec(tt::ARCH::QUASAR);
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}, {1, 0}));
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(fixture_device, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("has no producer or consumer")));
}

TEST_F(ProgramSpecTestQuasar, CPU_DFBPlacementUnset) {
    // Without an override, DFB placement contains all endpoint nodes.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeDFBPlacementSpec(tt::ARCH::QUASAR);
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    EXPECT_EQ(dfb->core_ranges, CoreRangeSet(CoreRange({0, 0}, {1, 0})));
}

TEST_F(ProgramSpecTestQuasar, CPU_DFBPlacementSuperset) {
    // An override that includes nodes without endpoints is rejected.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeDFBPlacementSpec(tt::ARCH::QUASAR);
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}, {2, 0}));
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(fixture_device, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("has no producer or consumer")));
}

TEST_F(ProgramSpecTestQuasar, CPU_DFBPlacementPartialOverlap) {
    // An override that includes nodes without endpoints is rejected.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeDFBPlacementSpec(tt::ARCH::QUASAR);
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({1, 0}, {2, 0}));
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(fixture_device, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("has no producer or consumer")));
}

TEST_F(ProgramSpecTestQuasar, CPU_DFBPlacementDisjoint) {
    // An override that includes nodes without endpoints is rejected.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeDFBPlacementSpec(tt::ARCH::QUASAR);
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({2, 0}));
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(fixture_device, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("has no producer or consumer")));
}

TEST_F(ProgramSpecTestQuasar, CPU_DFBPlacementPreservesKernelNodes) {
    // Restricting DFB storage to a subset of endpoint nodes preserves both kernels' execution nodes.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeDFBPlacementSpec(tt::ARCH::QUASAR);
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const CoreRangeSet execution_nodes(CoreRange({0, 0}, {1, 0}));
    EXPECT_EQ(program.impl().get_kernel_by_spec_name("dm_kernel")->core_range_set(), execution_nodes);
    EXPECT_EQ(program.impl().get_kernel_by_spec_name("compute_kernel")->core_range_set(), execution_nodes);
}

TYPED_TEST(DFBPlacementOverrideTest, CPU_DFBPlacementAliasesReserveStorageOnce) {
    // Aliases reserve one backing region on the selected endpoint node.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeDFBPlacementSpec(TypeParam::value, 3);
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    spec.dataflow_buffers[0].advanced_options.alias_with = {DFBSpecName{"dfb_1"}};
    spec.dataflow_buffers[1].advanced_options.alias_with = {DFBSpecName{"dfb_0"}};
    for (auto& dfb : spec.dataflow_buffers) {
        dfb.advanced_options.override_placement = CoreRangeSet(CoreRange({1, 0}));
    }
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto first = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    const auto next = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_2"));
    EXPECT_EQ(next->uniform_alloc_addr(), first->uniform_alloc_addr() + first->total_size());
}

TEST_F(ProgramSpecTestQuasar, CPU_DFBPlacementAliasesShareStorageWithDifferentEndpointNodes) {
    // Aliases with different inferred placements share storage on their selected common node.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeNestedQuasarDFBEndpointPlacementsForTest();
    spec.dataflow_buffers[0].advanced_options = {
        .alias_with = {DFBSpecName{"dfb_1"}}, .override_placement = CoreRangeSet(CoreRange({0, 0}))};
    spec.dataflow_buffers[1].advanced_options.alias_with = {DFBSpecName{"dfb_0"}};
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto first = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    const auto second = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_1"));
    EXPECT_EQ(first->uniform_alloc_addr(), second->uniform_alloc_addr());
}

TYPED_TEST(DFBPlacementOverrideTest, CPU_DFBPlacementReusesSlotsForDisjointEndpointNodes) {
    // Overrides retain slot reuse when the DFBs have disjoint inferred placements.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeSeparateDFBEndpointPairs(TypeParam::value);
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    spec.work_units[1].target_nodes = NodeRange({2, 0}, {3, 0});
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}));
    spec.dataflow_buffers[1].advanced_options.override_placement = CoreRangeSet(CoreRange({2, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto first = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    const auto second = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_1"));
    EXPECT_EQ(first->device_slot, second->device_slot);
}

TYPED_TEST(DFBPlacementOverrideTest, CPU_DFBPlacementSeparatesSlotsForOverlappingReservations) {
    // DFBs use different slots when they reserve storage on the same endpoint node.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeDFBPlacementSpec(TypeParam::value, 2);
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    spec.dataflow_buffers[1].advanced_options.override_placement = CoreRangeSet(CoreRange({1, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto first = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    const auto second = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_1"));
    EXPECT_NE(first->device_slot, second->device_slot);
}

TYPED_TEST(DFBPlacementOverrideTest, CPU_DFBPlacementIncludedNodeMask) {
    // The launch mask enables the DFB on a reserved kernel node.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeDFBPlacementSpec(TypeParam::value);
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    test_helpers::FinalizeMockDFBConfig(program);
    const auto* group = program.impl().kernels_on_core({0, 0}, test_helpers::TensixIndexForTest(program));
    EXPECT_EQ(group->launch_msg.view().kernel_config().local_cb_mask(), 1u);
}

TYPED_TEST(DFBPlacementOverrideTest, CPU_DFBPlacementPreservesGroupsWhenUnset) {
    // Nodes with identical kernels share one launch group when DFB placement is inferred.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeDFBPlacementSpec(TypeParam::value);
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto& groups = program.impl().get_kernel_groups(test_helpers::TensixIndexForTest(program));
    ASSERT_EQ(groups.size(), 1u);
    EXPECT_EQ(groups[0]->core_ranges, CoreRangeSet(CoreRange({0, 0}, {1, 0})));
}

TYPED_TEST(DFBPlacementOverrideTest, CPU_DFBPlacementDoesNotAllocateOnExcludedNodes) {
    // DFBs with disjoint backing placements can use the same L1 address.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeDisjointDFBPlacementsForTest(TypeParam::value);
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto first = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    const auto second = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_1"));
    EXPECT_EQ(first->uniform_alloc_addr(), second->uniform_alloc_addr());
}

TEST_F(ProgramSpecTestQuasar, CPU_DFBPlacementPreservesSize) {
    // Restricting reservation nodes preserves the DFB entry size and entry count.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeDFBPlacementSpec(tt::ARCH::QUASAR);
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    EXPECT_EQ(dfb->config.entry_size, spec.dataflow_buffers[0].entry_size);
    EXPECT_EQ(dfb->config.num_entries, spec.dataflow_buffers[0].num_entries);
}

TYPED_TEST(DFBPlacementOverrideTest, CPU_DFBPlacementAvoidsScratchpadCollision) {
    // A scratchpad starts after the DFB allocation on a node reserved by both resources.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeDFBPlacementSpec(TypeParam::value);
    const CoreRangeSet reserved(CoreRange({0, 0}));
    spec.dataflow_buffers[0].advanced_options.override_placement = reserved;
    spec.scratchpads = {{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = 64}};
    spec.scratchpads[0].advanced_options.override_placement = reserved;
    spec.kernels[0].scratchpad_bindings = {{ScratchpadSpecName{"pad"}, "pad"}};
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    const auto kernel = program.impl().get_kernel_by_spec_name("dm_kernel");
    EXPECT_GE(kernel->scratchpad_binding_handles()[0].allocated_address, dfb->uniform_alloc_addr() + dfb->total_size());
}

TYPED_TEST(DFBPlacementOverrideTest, CPU_DFBPlacementLeavesExcludedMemoryForScratchpad) {
    // A scratchpad on an excluded node can use the DFB's L1 address.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeDFBPlacementSpec(TypeParam::value);
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}));
    spec.scratchpads = {{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = 64}};
    spec.scratchpads[0].advanced_options.override_placement = CoreRangeSet(CoreRange({1, 0}));
    spec.kernels[0].scratchpad_bindings = {{ScratchpadSpecName{"pad"}, "pad"}};
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    const auto kernel = program.impl().get_kernel_by_spec_name("dm_kernel");
    EXPECT_EQ(kernel->scratchpad_binding_handles()[0].allocated_address, dfb->uniform_alloc_addr());
}

}  // namespace
}  // namespace tt::tt_metal::experimental
