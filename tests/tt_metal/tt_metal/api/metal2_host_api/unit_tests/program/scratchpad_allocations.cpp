// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestGen1;
using test_helpers::ProgramSpecTestQuasar;

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideUsesKernelPlacementWhenUnset) {
    // The binding records resolved placement without changing the kernel's execution nodes.
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
    // The binding records resolved placement without changing the kernel's execution nodes.
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
    // The binding records resolved placement without changing the kernel's execution nodes.
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
    // The binding records resolved placement without changing the kernel's execution nodes.
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
    // The binding records resolved placement without changing the kernel's execution nodes.
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

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideAllocatesOnceAndDeliversEveryBinding) {
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
        ScratchpadSpec{.unique_id = ScratchpadSpecName{"excluded_probe"}, .size_per_node = 64},
        ScratchpadSpec{.unique_id = ScratchpadSpecName{"extra_probe"}, .size_per_node = 64}};
    spec.work_units = {
        MakeMinimalWorkUnit("a", NodeCoord{0, 0}, {"a"}),
        MakeMinimalWorkUnit("b", NodeCoord{1, 0}, {"b"}),
        MakeMinimalWorkUnit("c", NodeCoord{3, 0}, {"c"}),
        MakeMinimalWorkUnit("extra", NodeCoord{2, 0}, {"probe_extra"})};

    auto program = MakeProgramFromSpec(*mesh_device_, spec);
    // Mock creation can skip JIT/allocation; explicitly exercise the host allocator.
    program.impl().allocate_scratchpads(mesh_device_.get());
    auto a = program.impl().get_kernel_by_spec_name("a");
    auto b = program.impl().get_kernel_by_spec_name("b");
    auto c = program.impl().get_kernel_by_spec_name("c");
    auto extra = program.impl().get_kernel_by_spec_name("probe_extra");
    const auto& shared = a->scratchpad_binding_handles()[0];
    ASSERT_TRUE(shared.shared_allocation_identifier.has_value());
    ASSERT_NE(shared.allocated_address, 0u);
    EXPECT_EQ(b->scratchpad_binding_handles()[0].shared_allocation_identifier, shared.shared_allocation_identifier);
    EXPECT_EQ(c->scratchpad_binding_handles()[0].shared_allocation_identifier, shared.shared_allocation_identifier);
    // Main allocates private bindings before shared groups. The override must account
    // for the private probe on its extra node, even though neither shared binder runs there.
    EXPECT_GE(shared.allocated_address, extra->scratchpad_binding_handles()[0].allocated_address + 64);
    const uint32_t original_address = shared.allocated_address;
    for (int repeat = 0; repeat < 2; ++repeat) {
        SetProgramRunArgs(program, {});
        for (const auto& kernel : {a, b, c}) {
            const auto& handle = kernel->scratchpad_binding_handles()[0];
            EXPECT_EQ(handle.allocated_address, shared.allocated_address);
            EXPECT_EQ(kernel->common_runtime_args()[handle.addr_crta_word], shared.allocated_address);
        }
    }
    // A fresh allocation cycle must recompute the record and patch every binding again.
    program.impl().invalidate_dataflow_buffer_allocation();
    program.impl().prepare_ephemeral_allocation(mesh_device_.get());
    program.impl().allocate_scratchpads(mesh_device_.get());
    EXPECT_EQ(a->scratchpad_binding_handles()[0].allocated_address, original_address);
    EXPECT_GE(shared.allocated_address, extra->scratchpad_binding_handles()[0].allocated_address + 64);
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideAddressIsRepatchedAfterDFBResize) {
    auto spec = test_helpers::MakeMinimalValidProgramSpec();
    spec.scratchpads = {ScratchpadSpec{
        .unique_id = ScratchpadSpecName{"pad"},
        .size_per_node = 64,
        .advanced_options = {.override_placement = CoreRangeSet(CoreRange({0, 0}, {1, 0}))}}};
    spec.kernels[0].scratchpad_bindings = {{ScratchpadSpecName{"pad"}, "pad"}};
    auto second = MakeMinimalGen2DMKernel("second");
    second.scratchpad_bindings = {{ScratchpadSpecName{"pad"}, "pad"}};
    spec.kernels.push_back(second);
    spec.work_units.push_back(MakeMinimalWorkUnit("second", NodeCoord{1, 0}, {"second"}));
    auto program = MakeProgramFromSpec(*mesh_device_, spec);
    auto first = program.impl().get_kernel_by_spec_name("dm_kernel");
    auto other = program.impl().get_kernel_by_spec_name("second");
    const uint32_t before = first->scratchpad_binding_handles()[0].allocated_address;
    ASSERT_NE(before, 0u);

    ProgramRunArgs args;
    args.dfb_run_overrides = {{.dfb = DFBSpecName{"dfb_0"}, .num_entries = 128}};
    SetProgramRunArgs(program, args);
    program.impl().prepare_ephemeral_allocation(mesh_device_.get());
    program.impl().allocate_dataflow_buffers(mesh_device_.get());
    program.impl().allocate_scratchpads(mesh_device_.get());
    const uint32_t after = first->scratchpad_binding_handles()[0].allocated_address;
    EXPECT_GT(after, before);
    for (const auto& kernel : {first, other}) {
        const auto& handle = kernel->scratchpad_binding_handles()[0];
        EXPECT_EQ(handle.allocated_address, after);
        EXPECT_EQ(kernel->common_runtime_args()[handle.addr_crta_word], after);
    }
    SetProgramRunArgs(program, {});
    for (const auto& kernel : {first, other}) {
        const auto& handle = kernel->scratchpad_binding_handles()[0];
        EXPECT_EQ(kernel->common_runtime_args()[handle.addr_crta_word], after);
    }
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideValidatesL1OnReservationOnlyNodes) {
    auto kernel = MakeMinimalGen2DMKernel("kernel");
    kernel.scratchpad_bindings = {{ScratchpadSpecName{"pad"}, "pad"}};
    ProgramSpec spec{
        .kernels = {kernel},
        .scratchpads = {ScratchpadSpec{
            .unique_id = ScratchpadSpecName{"pad"},
            .size_per_node = mesh_device_->l1_size_per_core(),
            .advanced_options = {.override_placement = CoreRangeSet(CoreRange({1, 0}))}}},
        .work_units = {MakeMinimalWorkUnit("work", NodeCoord{0, 0}, {"kernel"})}};
    // The program-local region starts above zero, so reserving all of L1 cannot fit.
    EXPECT_THROW(MakeProgramFromSpec(*mesh_device_, spec), std::runtime_error);
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverrideAndMultiBindingAreIndependent) {
    ProgramSpec spec;
    spec.kernels = {MakeMinimalGen2DMKernel("a"), MakeMinimalGen2DMKernel("b")};
    for (auto& kernel : spec.kernels) {
        kernel.scratchpad_bindings = {{ScratchpadSpecName{"pad"}, "pad"}};
    }
    const CoreRangeSet reserved(CoreRange({0, 0}, {1, 0}));
    spec.scratchpads = {ScratchpadSpec{
        .unique_id = ScratchpadSpecName{"pad"},
        .size_per_node = 64,
        .advanced_options = {.override_placement = reserved}}};
    spec.work_units = {MakeMinimalWorkUnit("work", NodeCoord{0, 0}, {"a", "b"})};
    // An override groups storage but does not grant permission to share a node.
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("allow_instance_multi_binding")));

    spec.scratchpads[0].advanced_options.allow_instance_multi_binding = true;
    auto program = MakeProgramFromSpec(*mesh_device_, spec);
    const auto a = program.impl().get_kernel_by_spec_name("a");
    const auto b = program.impl().get_kernel_by_spec_name("b");
    const auto& handle = a->scratchpad_binding_handles()[0];
    EXPECT_EQ(handle.placement, reserved);
    ASSERT_NE(handle.allocated_address, 0u);
    EXPECT_EQ(handle.allocated_address, b->scratchpad_binding_handles()[0].allocated_address);
    EXPECT_EQ(b->common_runtime_args()[handle.addr_crta_word], handle.allocated_address);
}

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadOverridePreservesGen2SharingRestrictions) {
    auto spec = test_helpers::MakeMinimalValidProgramSpec();
    for (auto& kernel : spec.kernels) {
        kernel.scratchpad_bindings = {{ScratchpadSpecName{"pad"}, "pad"}};
    }
    spec.scratchpads = {ScratchpadSpec{
        .unique_id = ScratchpadSpecName{"pad"},
        .size_per_node = 64,
        .advanced_options = {
            .allow_instance_multi_binding = true, .override_placement = CoreRangeSet(CoreRange({0, 0}))}}};
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("only between DM kernels on Gen2")));
    // Excluded nodes cannot access the scratchpad, so their binders do not share an instance.
    spec.scratchpads[0].advanced_options.override_placement = CoreRangeSet(CoreRange({1, 0}));
    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestGen1, CPU_ScratchpadOverrideSharedDMAndCompute) {
    auto spec = test_helpers::MakeMinimalGen1ValidProgramSpec();
    for (auto& kernel : spec.kernels) {
        kernel.scratchpad_bindings = {{ScratchpadSpecName{"pad"}, "pad"}};
    }
    spec.scratchpads = {ScratchpadSpec{
        .unique_id = ScratchpadSpecName{"pad"},
        .size_per_node = 64,
        .advanced_options = {
            .allow_instance_multi_binding = true, .override_placement = CoreRangeSet(CoreRange({0, 0}, {1, 0}))}}};
    auto program = MakeProgramFromSpec(*mesh_device_, spec);
    const auto a = program.impl().get_kernel_by_spec_name("dm_kernel");
    const auto b = program.impl().get_kernel_by_spec_name("compute_kernel");
    EXPECT_EQ(
        a->scratchpad_binding_handles()[0].allocated_address, b->scratchpad_binding_handles()[0].allocated_address);
}

}  // namespace
}  // namespace tt::tt_metal::experimental
