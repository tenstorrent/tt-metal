// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <stdexcept>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::test_helpers {

ProgramSpec MakeUnplacedScratchpadBinderSpecForTest(tt::ARCH arch = tt::ARCH::QUASAR) {
    auto binder = arch == tt::ARCH::QUASAR ? MakeMinimalGen2DMKernel("binder") : MakeMinimalGen1DMKernel("binder");
    binder.scratchpad_bindings = {{ScratchpadSpecName{"pad"}, "pad"}};
    return ProgramSpec{
        .kernels = {binder}, .scratchpads = {{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = 64}}};
}

}  // namespace tt::tt_metal::test_helpers

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalWorkUnit;
using test_helpers::MakeUnplacedScratchpadBinderSpecForTest;
using test_helpers::ProgramSpecTestBlackhole;
using test_helpers::ProgramSpecTestGen1;
using test_helpers::ProgramSpecTestQuasar;

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideAcceptsSubsetOfDerivedPlacement) {
    // A scratchpad can reserve storage on fewer nodes than its bound kernel uses.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = MakeUnplacedScratchpadBinderSpecForTest();
    spec.work_units = {MakeMinimalWorkUnit("work", NodeRange({0, 0}, {1, 0}), {"binder"})};
    spec.scratchpads[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}));
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideAcceptsEqualDerivedPlacement) {
    // An override can select every node in the scratchpad's derived placement.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = MakeUnplacedScratchpadBinderSpecForTest();
    spec.work_units = {MakeMinimalWorkUnit("work", NodeRange({0, 0}, {1, 0}), {"binder"})};
    spec.scratchpads[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}, {1, 0}));
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideAcceptsSubsetAcrossBoundKernels) {
    // An override can select nodes from different bound kernels without overlapping instances.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = MakeUnplacedScratchpadBinderSpecForTest();
    spec.kernels.push_back(spec.kernels[0]);
    spec.kernels[1].unique_id = KernelSpecName{"second"};
    spec.work_units = {
        MakeMinimalWorkUnit("first", NodeRange({0, 0}, {1, 0}), {"binder"}),
        MakeMinimalWorkUnit("second", NodeRange({2, 0}, {3, 0}), {"second"})};
    spec.scratchpads[0].advanced_options.override_placement =
        CoreRangeSet(std::vector<CoreRange>{CoreRange({0, 0}), CoreRange({2, 0})});
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideAcceptsNodesFromEveryBinderWorkUnit) {
    // All work units containing a bound kernel contribute to the scratchpad's derived placement.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = MakeUnplacedScratchpadBinderSpecForTest();
    spec.work_units = {
        MakeMinimalWorkUnit("first", NodeRange({0, 0}, {1, 0}), {"binder"}),
        MakeMinimalWorkUnit("second", NodeRange({2, 0}, {3, 0}), {"binder"})};
    spec.scratchpads[0].advanced_options.override_placement =
        CoreRangeSet(std::vector<CoreRange>{CoreRange({1, 0}), CoreRange({3, 0})});
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideAcceptsEqualNodesInSplitRanges) {
    // Splitting an override into separate ranges does not change which nodes it selects.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = MakeUnplacedScratchpadBinderSpecForTest();
    spec.work_units = {MakeMinimalWorkUnit("work", NodeRange({0, 0}, {1, 0}), {"binder"})};
    spec.scratchpads[0].advanced_options.override_placement =
        CoreRangeSet(std::vector<CoreRange>{CoreRange({0, 0}), CoreRange({1, 0})});
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideAcceptsNonContiguousSubset) {
    // Selected scratchpad nodes can have gaps within a bound kernel's placement.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = MakeUnplacedScratchpadBinderSpecForTest();
    spec.work_units = {MakeMinimalWorkUnit("work", NodeRange({0, 0}, {2, 0}), {"binder"})};
    spec.scratchpads[0].advanced_options.override_placement =
        CoreRangeSet(std::vector<CoreRange>{CoreRange({0, 0}), CoreRange({2, 0})});
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideAcceptsNonContiguousDerivedPlacement) {
    // An override can equal a derived placement whose nodes have gaps.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = MakeUnplacedScratchpadBinderSpecForTest();
    const CoreRangeSet derived(std::vector<CoreRange>{CoreRange({0, 0}), CoreRange({2, 0})});
    spec.work_units = {MakeMinimalWorkUnit("work", derived, {"binder"})};
    spec.scratchpads[0].advanced_options.override_placement = derived;
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideRejectsSupersetOfDerivedPlacement) {
    // An override cannot add nodes to the scratchpad's derived placement.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = MakeUnplacedScratchpadBinderSpecForTest();
    spec.work_units = {MakeMinimalWorkUnit("work", NodeRange({0, 0}, {1, 0}), {"binder"})};
    spec.scratchpads[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}, {2, 0}));
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(fixture_device, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("ScratchpadSpec 'pad' override_placement must be a subset")));
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideRejectsMixedDerivedAndOutsideNodes) {
    // Omitting a derived node does not permit selecting a node outside the derived placement.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = MakeUnplacedScratchpadBinderSpecForTest();
    spec.work_units = {MakeMinimalWorkUnit("work", NodeRange({0, 0}, {1, 0}), {"binder"})};
    spec.scratchpads[0].advanced_options.override_placement = CoreRangeSet(CoreRange({1, 0}, {2, 0}));
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(fixture_device, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("ScratchpadSpec 'pad' override_placement must be a subset")));
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideRejectsOnlyOutsideNodes) {
    // Scratchpad storage cannot be placed entirely outside its derived placement.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = MakeUnplacedScratchpadBinderSpecForTest();
    spec.work_units = {MakeMinimalWorkUnit("work", NodeRange({0, 0}, {1, 0}), {"binder"})};
    spec.scratchpads[0].advanced_options.override_placement = CoreRangeSet(CoreRange({2, 0}));
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(fixture_device, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("ScratchpadSpec 'pad' override_placement must be a subset")));
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideRejectsNodesOfUnboundKernel) {
    // A kernel that does not bind the scratchpad cannot contribute nodes to its derived placement.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = MakeUnplacedScratchpadBinderSpecForTest();
    spec.kernels.push_back(test_helpers::MakeMinimalGen2DMKernel("unrelated"));
    spec.work_units = {
        MakeMinimalWorkUnit("bound", NodeCoord{0, 0}, {"binder"}),
        MakeMinimalWorkUnit("unbound", NodeCoord{1, 0}, {"unrelated"})};
    spec.scratchpads[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}, {1, 0}));
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(fixture_device, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("ScratchpadSpec 'pad' override_placement must be a subset")));
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideRejectsRangeContainingOutsideNode) {
    // A range cannot include the unplaced node between two derived nodes.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = MakeUnplacedScratchpadBinderSpecForTest();
    const CoreRangeSet derived(std::vector<CoreRange>{CoreRange({0, 0}), CoreRange({2, 0})});
    spec.work_units = {MakeMinimalWorkUnit("work", derived, {"binder"})};
    spec.scratchpads[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}, {2, 0}));
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(fixture_device, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("ScratchpadSpec 'pad' override_placement must be a subset")));
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideRejectsOutsideNodesWithSharingEnabled) {
    // Quasar sharing permission does not permit storage outside the bound kernels' nodes.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = MakeUnplacedScratchpadBinderSpecForTest();
    spec.kernels.push_back(spec.kernels[0]);
    spec.kernels[1].unique_id = KernelSpecName{"second"};
    spec.work_units = {MakeMinimalWorkUnit("work", NodeRange({0, 0}, {1, 0}), {"binder", "second"})};
    spec.scratchpads[0].advanced_options = {
        .allow_instance_multi_binding = true, .override_placement = CoreRangeSet(CoreRange({0, 0}, {2, 0}))};
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(fixture_device, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("ScratchpadSpec 'pad' override_placement must be a subset")));
}

TEST_F(ProgramSpecTestGen1, CPU_ScratchpadOverrideRejectsOutsideNodesWithSharingEnabled) {
    // Wormhole sharing permission does not permit storage outside the bound kernels' nodes.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = MakeUnplacedScratchpadBinderSpecForTest(tt::ARCH::WORMHOLE_B0);
    spec.kernels.push_back(test_helpers::MakeMinimalGen1DMKernel("second", DataMovementProcessor::RISCV_1));
    spec.kernels[1].scratchpad_bindings = spec.kernels[0].scratchpad_bindings;
    spec.work_units = {MakeMinimalWorkUnit("work", NodeRange({0, 0}, {1, 0}), {"binder", "second"})};
    spec.scratchpads[0].advanced_options = {
        .allow_instance_multi_binding = true, .override_placement = CoreRangeSet(CoreRange({0, 0}, {2, 0}))};
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(fixture_device, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("ScratchpadSpec 'pad' override_placement must be a subset")));
}

TEST_F(ProgramSpecTestBlackhole, CPU_ScratchpadOverrideRejectsOutsideNodesWithSharingEnabled) {
    // Blackhole sharing permission does not permit storage outside the bound kernels' nodes.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = MakeUnplacedScratchpadBinderSpecForTest(tt::ARCH::BLACKHOLE);
    spec.kernels.push_back(test_helpers::MakeMinimalGen1DMKernel("second", DataMovementProcessor::RISCV_1));
    spec.kernels[1].scratchpad_bindings = spec.kernels[0].scratchpad_bindings;
    spec.work_units = {MakeMinimalWorkUnit("work", NodeRange({0, 0}, {1, 0}), {"binder", "second"})};
    spec.scratchpads[0].advanced_options = {
        .allow_instance_multi_binding = true, .override_placement = CoreRangeSet(CoreRange({0, 0}, {2, 0}))};
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(fixture_device, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("ScratchpadSpec 'pad' override_placement must be a subset")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
