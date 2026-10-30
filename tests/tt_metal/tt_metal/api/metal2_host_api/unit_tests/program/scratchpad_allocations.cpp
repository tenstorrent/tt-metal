// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "impl/allocator/allocator.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace test_helpers {

ProgramSpec MakeScratchpadOverrideWithProbesForTest() {
    ProgramSpec spec;
    // a and c access the shared allocation; b binds it on an excluded node.
    for (const auto& name : {"a", "b", "c", "probe_extra"}) {
        spec.kernels.push_back(MakeMinimalGen2DMKernel(name));
    }
    for (size_t i = 0; i < 3; ++i) {
        spec.kernels[i].scratchpad_bindings = {{ScratchpadSpecName{"shared"}, "shared"}};
    }
    spec.kernels[1].scratchpad_bindings.push_back({ScratchpadSpecName{"excluded_probe"}, "probe"});
    spec.kernels[3].scratchpad_bindings = {{ScratchpadSpecName{"extra_probe"}, "probe"}};
    spec.scratchpads = {
        ScratchpadSpec{
            .unique_id = ScratchpadSpecName{"shared"},
            .size_per_node = 1024,
            .advanced_options =
                {.override_placement =
                     CoreRangeSet(std::vector<CoreRange>{CoreRange({0, 0}), CoreRange({2, 0}, {4, 0})})}},
        ScratchpadSpec{.unique_id = ScratchpadSpecName{"excluded_probe"}, .size_per_node = 4096},
        ScratchpadSpec{.unique_id = ScratchpadSpecName{"extra_probe"}, .size_per_node = 64}};
    spec.work_units = {
        MakeMinimalWorkUnit("a", NodeCoord{0, 0}, {"a"}),
        MakeMinimalWorkUnit("b", NodeCoord{1, 0}, {"b"}),
        MakeMinimalWorkUnit("c", NodeCoord{3, 0}, {"c"}),
        MakeMinimalWorkUnit("extra", NodeCoord{2, 0}, {"probe_extra"})};

    return spec;
}

ProgramSpec MakeTwoScratchpadBindersForTest(tt::ARCH arch, bool include_compute = false) {
    auto spec = arch == tt::ARCH::QUASAR ? MakeMinimalValidProgramSpec() : MakeMinimalGen1ValidProgramSpec();
    spec.dataflow_buffers.clear();
    if (!include_compute) {
        spec.kernels[1] = arch == tt::ARCH::QUASAR ? MakeMinimalGen2DMKernel("second")
                                                   : MakeMinimalGen1DMKernel("second", DataMovementProcessor::RISCV_1);
    }
    for (auto& kernel : spec.kernels) {
        kernel.dfb_bindings.clear();
        kernel.scratchpad_bindings = {{ScratchpadSpecName{"pad"}, "pad"}};
    }
    spec.scratchpads = {{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = 64}};
    spec.work_units = {MakeMinimalWorkUnit("binders", NodeCoord{0, 0}, {})};
    spec.work_units[0].kernels = {spec.kernels[0].unique_id, spec.kernels[1].unique_id};
    return spec;
}

ProgramSpec MakeScratchpadWithResizableDFBForTest() {
    auto spec = MakeMinimalValidProgramSpec();
    spec.scratchpads = {{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = 64}};
    spec.scratchpads[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}, {1, 0}));
    spec.kernels[0].scratchpad_bindings = {{ScratchpadSpecName{"pad"}, "pad"}};
    auto second = MakeMinimalGen2DMKernel("second");
    second.scratchpad_bindings = {{ScratchpadSpecName{"pad"}, "pad"}};
    spec.kernels.push_back(second);
    spec.work_units.push_back(MakeMinimalWorkUnit("second", NodeCoord{1, 0}, {"second"}));
    return spec;
}

const ScratchpadBindingHandle& ScratchpadBindingForTest(const Program& program, const std::string& kernel) {
    return program.impl().get_kernel_by_spec_name(kernel)->scratchpad_binding_handles()[0];
}

}  // namespace test_helpers
namespace {

using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestGen1;
using test_helpers::ProgramSpecTestQuasar;

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideUsesKernelPlacementWhenUnset) {
    // An unset override allocates the scratchpad on its kernel nodes.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeMinimalValidProgramSpec();
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = 64}};
    spec.kernels[0].scratchpad_bindings = {{ScratchpadSpecName{"pad"}, "pad"}};
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto kernel = program.impl().get_kernel_by_spec_name("dm_kernel");
    EXPECT_EQ(kernel->scratchpad_binding_handles()[0].placement, kernel->core_range_set());
    EXPECT_EQ(kernel->core_range_set(), CoreRangeSet(CoreRange({0, 0}, {1, 0})));
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideResolvesSubset) {
    // A subset override reserves only the selected kernel node.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeMinimalValidProgramSpec();
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = 64}};
    spec.kernels[0].scratchpad_bindings = {{ScratchpadSpecName{"pad"}, "pad"}};
    spec.scratchpads[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto kernel = program.impl().get_kernel_by_spec_name("dm_kernel");
    EXPECT_EQ(kernel->scratchpad_binding_handles()[0].placement, CoreRangeSet(CoreRange({0, 0})));
    EXPECT_EQ(kernel->core_range_set(), CoreRangeSet(CoreRange({0, 0}, {1, 0})));
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideResolvesSuperset) {
    // A superset override also reserves nodes outside the kernel placement.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeMinimalValidProgramSpec();
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = 64}};
    spec.kernels[0].scratchpad_bindings = {{ScratchpadSpecName{"pad"}, "pad"}};
    spec.scratchpads[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}, {2, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto kernel = program.impl().get_kernel_by_spec_name("dm_kernel");
    EXPECT_EQ(kernel->scratchpad_binding_handles()[0].placement, CoreRangeSet(CoreRange({0, 0}, {2, 0})));
    EXPECT_EQ(kernel->core_range_set(), CoreRangeSet(CoreRange({0, 0}, {1, 0})));
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideResolvesPartialOverlap) {
    // A partial override omits a kernel node and reserves an extra node.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeMinimalValidProgramSpec();
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = 64}};
    spec.kernels[0].scratchpad_bindings = {{ScratchpadSpecName{"pad"}, "pad"}};
    spec.scratchpads[0].advanced_options.override_placement = CoreRangeSet(CoreRange({1, 0}, {2, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto kernel = program.impl().get_kernel_by_spec_name("dm_kernel");
    EXPECT_EQ(kernel->scratchpad_binding_handles()[0].placement, CoreRangeSet(CoreRange({1, 0}, {2, 0})));
    EXPECT_EQ(kernel->core_range_set(), CoreRangeSet(CoreRange({0, 0}, {1, 0})));
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideResolvesDisjointPlacement) {
    // A disjoint override reserves storage only outside the kernel placement.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeMinimalValidProgramSpec();
    spec.work_units[0].target_nodes = NodeRange({0, 0}, {1, 0});
    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = 64}};
    spec.kernels[0].scratchpad_bindings = {{ScratchpadSpecName{"pad"}, "pad"}};
    spec.scratchpads[0].advanced_options.override_placement = CoreRangeSet(CoreRange({2, 0}));
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto kernel = program.impl().get_kernel_by_spec_name("dm_kernel");
    EXPECT_EQ(kernel->scratchpad_binding_handles()[0].placement, CoreRangeSet(CoreRange({2, 0})));
    EXPECT_EQ(kernel->core_range_set(), CoreRangeSet(CoreRange({0, 0}, {1, 0})));
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideAllocatesOnceForAllBindings) {
    // Every binding sees one shared scratchpad address, including a binding on an excluded node.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeScratchpadOverrideWithProbesForTest();
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto& first = test_helpers::ScratchpadBindingForTest(program, "a");
    EXPECT_NE(first.allocated_address, 0u);
    EXPECT_EQ(test_helpers::ScratchpadBindingForTest(program, "b").allocated_address, first.allocated_address);
    EXPECT_EQ(test_helpers::ScratchpadBindingForTest(program, "c").allocated_address, first.allocated_address);
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideAvoidsExtraNodeAllocation) {
    // An overridden scratchpad does not overlap another scratchpad on an extra reserved node.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeScratchpadOverrideWithProbesForTest();
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto& shared = test_helpers::ScratchpadBindingForTest(program, "a");
    const auto& extra = test_helpers::ScratchpadBindingForTest(program, "probe_extra");
    EXPECT_GE(shared.allocated_address, extra.allocated_address + extra.size_bytes);
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideDoesNotReserveExcludedNodes) {
    // Storage on an excluded node does not raise the overridden scratchpad's allocation address.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeScratchpadOverrideWithProbesForTest();
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto& shared = test_helpers::ScratchpadBindingForTest(program, "a");
    const auto& excluded = program.impl().get_kernel_by_spec_name("b")->scratchpad_binding_handles()[1];
    EXPECT_LT(shared.allocated_address, excluded.allocated_address + excluded.size_bytes);
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideRepeatedArgsDeliverEveryBinding) {
    // Repeated runtime argument updates keep the shared address available to every binding.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeScratchpadOverrideWithProbesForTest();
    auto program = MakeProgramFromSpec(fixture_device, spec);
    SetProgramRunArgs(program, {});
    SetProgramRunArgs(program, {});
    const auto address = test_helpers::ScratchpadBindingForTest(program, "a").allocated_address;
    for (const auto* name : {"a", "b", "c"}) {
        const auto kernel = program.impl().get_kernel_by_spec_name(name);
        const auto& binding = kernel->scratchpad_binding_handles()[0];
        EXPECT_EQ(kernel->common_runtime_args()[binding.addr_crta_word], address);
    }
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideReallocationPreservesAddress) {
    // Reallocating an unchanged program preserves the shared scratchpad address.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeScratchpadOverrideWithProbesForTest();
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto address = test_helpers::ScratchpadBindingForTest(program, "a").allocated_address;
    program.impl().invalidate_dataflow_buffer_allocation();
    program.impl().compile_and_allocate(&fixture_device, false);
    EXPECT_EQ(test_helpers::ScratchpadBindingForTest(program, "a").allocated_address, address);
    EXPECT_EQ(test_helpers::ScratchpadBindingForTest(program, "c").allocated_address, address);
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideAddressIsRepatchedAfterDFBResize) {
    // Growing a DFB moves the scratchpad and supplies the new address to every binding.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeScratchpadWithResizableDFBForTest();
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto before = test_helpers::ScratchpadBindingForTest(program, "dm_kernel").allocated_address;
    ProgramRunArgs args{.dfb_run_overrides = {{.dfb = DFBSpecName{"dfb_0"}, .num_entries = 128}}};
    SetProgramRunArgs(program, args);
    program.impl().compile_and_allocate(&fixture_device, false);
    const auto& first = test_helpers::ScratchpadBindingForTest(program, "dm_kernel");
    const auto second = program.impl().get_kernel_by_spec_name("second");
    EXPECT_GT(first.allocated_address, before);
    EXPECT_EQ(second->scratchpad_binding_handles()[0].allocated_address, first.allocated_address);
    EXPECT_EQ(second->common_runtime_args()[first.addr_crta_word], first.allocated_address);
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideValidatesL1OnReservationOnlyNodes) {
    // A scratchpad that fits on its kernel node is rejected when persistent L1 fills its reservation node.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto& arena = fixture_device.allocator_impl()->persistent_l1();
    const CoreRangeSet reserved(CoreRange({2, 0}));
    const auto persistent = arena.allocate(reserved, 65536, 32);
    auto spec = test_helpers::MakeTwoScratchpadBindersForTest(tt::ARCH::QUASAR);
    spec.scratchpads[0].size_per_node = fixture_device.l1_size_per_core() - persistent.address - 32;
    spec.scratchpads[0].advanced_options.override_placement = reserved;
    spec.scratchpads[0].advanced_options.allow_instance_multi_binding = true;
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(fixture_device, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("beyond max L1 size")));
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideRequiresMultiBindingPermission) {
    // A placement override does not permit two kernels to share a scratchpad on the same node.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeTwoScratchpadBindersForTest(tt::ARCH::QUASAR);
    spec.scratchpads[0].advanced_options.override_placement = CoreRangeSet(CoreRange({0, 0}, {1, 0}));
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(fixture_device, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("allow_instance_multi_binding")));
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideAllowsSharedDMStorage) {
    // DM kernels can share an overridden scratchpad when multi-binding is enabled.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeTwoScratchpadBindersForTest(tt::ARCH::QUASAR);
    spec.scratchpads[0].advanced_options = {
        .allow_instance_multi_binding = true, .override_placement = CoreRangeSet(CoreRange({0, 0}, {1, 0}))};
    auto program = MakeProgramFromSpec(fixture_device, spec);
    const auto& first = test_helpers::ScratchpadBindingForTest(program, "dm_kernel");
    EXPECT_NE(first.allocated_address, 0u);
    EXPECT_EQ(test_helpers::ScratchpadBindingForTest(program, "second").allocated_address, first.allocated_address);
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverridePreservesGen2SharingRestrictions) {
    // Quasar rejects a scratchpad shared by DM and compute kernels on a reserved node.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeTwoScratchpadBindersForTest(tt::ARCH::QUASAR, true);
    spec.scratchpads[0].advanced_options = {
        .allow_instance_multi_binding = true, .override_placement = CoreRangeSet(CoreRange({0, 0}))};
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(fixture_device, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("only between DM kernels on Gen2")));
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideIgnoresExcludedSharing) {
    // Kernels on an excluded node do not share an instance of the scratchpad.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeTwoScratchpadBindersForTest(tt::ARCH::QUASAR, true);
    spec.scratchpads[0].advanced_options = {
        .allow_instance_multi_binding = true, .override_placement = CoreRangeSet(CoreRange({1, 0}))};
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device, spec));
}

TEST_F(ProgramSpecTestGen1, CPU_ScratchpadOverrideSharedDMAndCompute) {
    // Gen1 DM and compute kernels can share scratchpad storage on the overridden placement.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = test_helpers::MakeTwoScratchpadBindersForTest(tt::ARCH::WORMHOLE_B0, true);
    spec.scratchpads[0].advanced_options = {
        .allow_instance_multi_binding = true, .override_placement = CoreRangeSet(CoreRange({0, 0}, {1, 0}))};
    auto program = MakeProgramFromSpec(fixture_device, spec);
    EXPECT_EQ(
        test_helpers::ScratchpadBindingForTest(program, "dm_kernel").allocated_address,
        test_helpers::ScratchpadBindingForTest(program, "compute_kernel").allocated_address);
}

}  // namespace
}  // namespace tt::tt_metal::experimental
