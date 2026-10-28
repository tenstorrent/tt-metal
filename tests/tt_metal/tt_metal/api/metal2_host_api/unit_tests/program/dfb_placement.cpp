// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/dfb_placement_test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace test_helpers {

static uint32_t TensixIndex(const Program& program) {
    return MetalContext::instance(program.impl().get_context_id())
        .hal()
        .get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
}

}  // namespace test_helpers
namespace {

using test_helpers::ProgramSpecTestBlackhole;
using test_helpers::ProgramSpecTestQuasar;

template <typename Arch>
class DFBPlacementOverrideTest : public test_helpers::MockMeshDeviceFixture<Arch::value> {
protected:
    distributed::MeshDevice& fixture_device() { return *this->mesh_device_; }
};

using Architectures = ::testing::
    Types<std::integral_constant<tt::ARCH, tt::ARCH::WORMHOLE_B0>, std::integral_constant<tt::ARCH, tt::ARCH::QUASAR>>;
TYPED_TEST_SUITE(DFBPlacementOverrideTest, Architectures);

TEST_F(ProgramSpecTestQuasar, CPU_DFBPlacementUnset) {
    // Without an override, DFB placement contains all endpoint nodes.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeDFBPlacementSpec(tt::ARCH::QUASAR);
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    EXPECT_EQ(dfb->core_ranges, CoreRangeSet(CoreRange({0, 0}, {1, 0})));
}

TEST_F(ProgramSpecTestQuasar, CPU_DFBPlacementSubset) {
    // A subset override reserves only the selected endpoint nodes.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeDFBPlacementSpec(tt::ARCH::QUASAR);
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    EXPECT_EQ(dfb->core_ranges, CoreRangeSet(CoreRange({0, 0})));
}

TEST_F(ProgramSpecTestQuasar, CPU_DFBPlacementSuperset) {
    // A superset override reserves all endpoint nodes and the extra nodes.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeDFBPlacementSpec(tt::ARCH::QUASAR);
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}, {2, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    EXPECT_EQ(dfb->core_ranges, CoreRangeSet(CoreRange({0, 0}, {2, 0})));
}

TEST_F(ProgramSpecTestQuasar, CPU_DFBPlacementPartialOverlap) {
    // A partially overlapping override omits some endpoint nodes and reserves extra nodes.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeDFBPlacementSpec(tt::ARCH::QUASAR);
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({1, 0}, {2, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    EXPECT_EQ(dfb->core_ranges, CoreRangeSet(CoreRange({1, 0}, {2, 0})));
}

TEST_F(ProgramSpecTestQuasar, CPU_DFBPlacementDisjoint) {
    // A disjoint override reserves only nodes outside the endpoint placements.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeDFBPlacementSpec(tt::ARCH::QUASAR);
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({2, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    EXPECT_EQ(dfb->core_ranges, CoreRangeSet(CoreRange({2, 0})));
}

TEST_F(ProgramSpecTestQuasar, CPU_DFBPlacementPreservesKernelNodes) {
    // Moving DFB storage outside the endpoint nodes preserves both kernels' execution nodes.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeDFBPlacementSpec(tt::ARCH::QUASAR);
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({2, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const CoreRangeSet execution_nodes(CoreRange({0, 0}, {1, 0}));
    EXPECT_EQ(program.impl().get_kernel_by_spec_name("dm_kernel")->core_range_set(), execution_nodes);
    EXPECT_EQ(program.impl().get_kernel_by_spec_name("compute_kernel")->core_range_set(), execution_nodes);
}

TYPED_TEST(DFBPlacementOverrideTest, CPU_DFBPlacementAliasesUseEffectiveNodes) {
    // Aliased DFBs with different endpoint nodes share one L1 address on their common reserved nodes.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeSeparateDFBEndpointPairs(TypeParam::value);
    spec.dataflow_buffers[0].advanced_options.alias_with = {DFBSpecName{"dfb_1"}};
    spec.dataflow_buffers[1].advanced_options.alias_with = {DFBSpecName{"dfb_0"}};
    for (auto& dfb : spec.dataflow_buffers) {
        dfb.advanced_options.override_placement = CoreRangeSet(CoreRange({2, 0}));
    }
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto first = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    const auto second = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_1"));
    EXPECT_EQ(first->uniform_alloc_addr(), second->uniform_alloc_addr());
}

TYPED_TEST(DFBPlacementOverrideTest, CPU_DFBPlacementReusesSlotsForDisjointReservations) {
    // DFBs with disjoint reserved nodes reuse a device slot even when their endpoint kernels run together.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeDFBPlacementSpec(TypeParam::value, 2);
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}));
    spec.dataflow_buffers[1].advanced_options.override_placement = CoreRangeSet(CoreRange({1, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto first = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    const auto second = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_1"));
    EXPECT_EQ(first->device_slot, second->device_slot);
}

TYPED_TEST(DFBPlacementOverrideTest, CPU_DFBPlacementSeparatesSlotsForOverlappingReservations) {
    // DFBs with disjoint endpoint nodes use different slots when their reserved nodes overlap.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeSeparateDFBEndpointPairs(TypeParam::value);
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}, {2, 0}));
    spec.dataflow_buffers[1].advanced_options.override_placement = CoreRangeSet(CoreRange({1, 0}, {2, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto first = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    const auto second = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_1"));
    EXPECT_NE(first->device_slot, second->device_slot);
}

TYPED_TEST(DFBPlacementOverrideTest, CPU_DFBPlacementSplitsKernelGroups) {
    // Nodes with the same kernels form separate groups when only one node reserves the DFB.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeDFBPlacementSpec(TypeParam::value);
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const uint32_t index = test_helpers::TensixIndex(program);
    EXPECT_NE(program.impl().kernels_on_core({0, 0}, index), program.impl().kernels_on_core({1, 0}, index));
}

TYPED_TEST(DFBPlacementOverrideTest, CPU_DFBPlacementGroupsByDFBIdentity) {
    // Nodes with the same kernels form separate groups for distinct DFBs that reuse a device slot.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeDFBPlacementSpec(TypeParam::value, 2);
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}));
    spec.dataflow_buffers[1].advanced_options.override_placement = CoreRangeSet(CoreRange({1, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const uint32_t index = test_helpers::TensixIndex(program);
    EXPECT_NE(program.impl().kernels_on_core({0, 0}, index), program.impl().kernels_on_core({1, 0}, index));
}

TYPED_TEST(DFBPlacementOverrideTest, CPU_DFBPlacementExcludedNodeMask) {
    // The launch mask disables the DFB on an excluded kernel node.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeDFBPlacementSpec(TypeParam::value);
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    test_helpers::FinalizeMockDFBConfig(program);
    const auto* group = program.impl().kernels_on_core({1, 0}, test_helpers::TensixIndex(program));
    EXPECT_EQ(group->launch_msg.view().kernel_config().local_cb_mask(), 0u);
}

TYPED_TEST(DFBPlacementOverrideTest, CPU_DFBPlacementIncludedNodeMask) {
    // The launch mask enables the DFB on a reserved kernel node.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeDFBPlacementSpec(TypeParam::value);
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    test_helpers::FinalizeMockDFBConfig(program);
    const auto* group = program.impl().kernels_on_core({0, 0}, test_helpers::TensixIndex(program));
    EXPECT_EQ(group->launch_msg.view().kernel_config().local_cb_mask(), 1u);
}

TYPED_TEST(DFBPlacementOverrideTest, CPU_DFBPlacementDoesNotLaunchReservationOnlyNodes) {
    // Extra DFB reservation nodes do not expand the kernel launch group.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeDFBPlacementSpec(TypeParam::value);
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}, {2, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto& groups = program.impl().get_kernel_groups(test_helpers::TensixIndex(program));
    ASSERT_EQ(groups.size(), 1u);
    EXPECT_EQ(groups[0]->core_ranges, CoreRangeSet(CoreRange({0, 0})));
}

TYPED_TEST(DFBPlacementOverrideTest, CPU_DFBPlacementSizesConfigWithoutKernelOverlap) {
    // Configuration storage holds a DFB whose reserved nodes are outside all kernel placements.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeDFBPlacementSpec(TypeParam::value);
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({2, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    test_helpers::FinalizeMockDFBConfig(program);
    const auto& config = program.impl().get_program_config(test_helpers::TensixIndex(program));
    const auto dfbs = program.impl().dataflow_buffers_on_core({2, 0});
    ASSERT_EQ(dfbs.size(), 1u);
    std::vector<uint8_t> payload(config.dfb_size);
    EXPECT_GT(dfb::detail::serialize_dfb_config_for_core({2, 0}, dfbs, payload), 0u);
}

TYPED_TEST(DFBPlacementOverrideTest, CPU_DFBPlacementSizesConfigForSlotsOnlyOnExtraNodes) {
    // Configuration storage holds the higher DFB slot used only on an extra reservation node.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeDFBPlacementSpec(TypeParam::value, 2);
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}, {2, 0}));
    spec.dataflow_buffers[1].advanced_options.override_placement = CoreRangeSet(CoreRange({2, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    test_helpers::FinalizeMockDFBConfig(program);
    const auto& config = program.impl().get_program_config(test_helpers::TensixIndex(program));
    const auto dfbs = program.impl().dataflow_buffers_on_core({2, 0});
    ASSERT_EQ(dfbs.size(), 2u);
    std::vector<uint8_t> payload(config.dfb_size);
    EXPECT_GT(dfb::detail::serialize_dfb_config_for_core({2, 0}, dfbs, payload), 0u);
}

TYPED_TEST(DFBPlacementOverrideTest, CPU_DFBPlacementPreservesGroupsWhenUnset) {
    // Nodes with identical kernels share one launch group when DFB placement is inferred.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeDFBPlacementSpec(TypeParam::value);
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto& groups = program.impl().get_kernel_groups(test_helpers::TensixIndex(program));
    ASSERT_EQ(groups.size(), 1u);
    EXPECT_EQ(groups[0]->core_ranges, CoreRangeSet(CoreRange({0, 0}, {1, 0})));
}

TYPED_TEST(DFBPlacementOverrideTest, CPU_DFBPlacementAllocatesOnExtraNodes) {
    // An extra reservation node receives the same nonzero DFB address as an endpoint node.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeDFBPlacementSpec(TypeParam::value);
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}, {2, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    ASSERT_TRUE(dfb->core_lookup_.contains(CoreCoord(2, 0)));
    EXPECT_EQ(dfb->core_lookup_.at(CoreCoord(2, 0)).second, dfb->core_lookup_.at(CoreCoord(0, 0)).second);
    EXPECT_GT(dfb->core_lookup_.at(CoreCoord(2, 0)).second, 0u);
}

TYPED_TEST(DFBPlacementOverrideTest, CPU_DFBPlacementDoesNotAllocateOnExcludedNodes) {
    // An excluded kernel node has no DFB backing allocation.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeDFBPlacementSpec(TypeParam::value);
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    EXPECT_FALSE(dfb->core_lookup_.contains(CoreCoord(1, 0)));
}

TEST_F(ProgramSpecTestQuasar, CPU_DFBPlacementPreservesSize) {
    // Adding reservation nodes preserves the DFB entry size and entry count.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeDFBPlacementSpec(tt::ARCH::QUASAR);
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}, {2, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    EXPECT_EQ(dfb->config.entry_size, spec.dataflow_buffers[0].entry_size);
    EXPECT_EQ(dfb->config.num_entries, spec.dataflow_buffers[0].num_entries);
}

TYPED_TEST(DFBPlacementOverrideTest, CPU_DFBPlacementAvoidsScratchpadCollisionOnExtraNode) {
    // A scratchpad starts after the DFB allocation on a shared reservation node with no kernels.
    distributed::MeshDevice& fixture_device = this->fixture_device();
    auto spec = test_helpers::MakeDFBPlacementSpec(TypeParam::value);
    const CoreRangeSet reserved(CoreRange({2, 0}));
    spec.dataflow_buffers[0].advanced_options.override_placement = reserved;
    spec.scratchpads = {{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = 64}};
    spec.scratchpads[0].advanced_options.override_placement = reserved;
    spec.kernels[0].scratchpad_bindings = {{ScratchpadSpecName{"pad"}, "pad"}};
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    const auto kernel = program.impl().get_kernel_by_spec_name("dm_kernel");
    EXPECT_GE(kernel->scratchpad_binding_handles()[0].allocated_address, dfb->uniform_alloc_addr() + dfb->total_size());
}

TEST_F(ProgramSpecTestBlackhole, CPU_DFBPlacementSubset) {
    // A subset override reserves only the selected endpoint nodes.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeDFBPlacementSpec(tt::ARCH::BLACKHOLE);
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    EXPECT_EQ(dfb->core_ranges, CoreRangeSet(CoreRange({0, 0})));
}

TEST_F(ProgramSpecTestBlackhole, CPU_DFBPlacementSizesConfigWithoutKernelOverlap) {
    // Configuration storage holds a DFB whose reserved nodes are outside all kernel placements.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeDFBPlacementSpec(tt::ARCH::BLACKHOLE);
    spec.dataflow_buffers[0].advanced_options.override_placement = CoreRangeSet(CoreRange({2, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    test_helpers::FinalizeMockDFBConfig(program);
    const auto& config = program.impl().get_program_config(test_helpers::TensixIndex(program));
    const auto dfbs = program.impl().dataflow_buffers_on_core({2, 0});
    ASSERT_EQ(dfbs.size(), 1u);
    std::vector<uint8_t> payload(config.dfb_size);
    EXPECT_GT(dfb::detail::serialize_dfb_config_for_core({2, 0}, dfbs, payload), 0u);
}

}  // namespace
}  // namespace tt::tt_metal::experimental
