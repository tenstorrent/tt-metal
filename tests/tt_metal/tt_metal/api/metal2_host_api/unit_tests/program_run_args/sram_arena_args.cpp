// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <array>
#include <stdexcept>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/prefetcher_pipe.hpp>
#include <tt-metalium/experimental/sram_arena/sram_arena.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "impl/sram_arena/sram_arena_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::ProgramRunArgsTestGen1;

RangeLockstepSRAMArenaProperties AllocationProperties() {
    return {.banks_to_be_allocated_on = NodeRangeSet{NodeRange({0, 0}, {1, 0})}, .per_bank_size = 4096};
}

ProgramSpec BorrowedScratchpadSpec(bool shared = false) {
    ProgramSpec spec;
    spec.kernels = {test_helpers::MakeMinimalGen1DMKernel("writer")};
    spec.kernels[0].scratchpad_bindings = {{.scratchpad_spec_name = ScratchpadSpecName{"pad"}, .accessor_name = "pad"}};
    spec.scratchpads = {{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = 1024}};
    spec.scratchpads[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"arena"}, .offset = 1024};
    spec.advanced_options.sram_arena_parameters = {
        {.unique_id = SRAMArenaParamName{"arena"}, .properties = AllocationProperties()}};
    spec.work_units = {
        test_helpers::MakeMinimalWorkUnit("work", AllocationProperties().banks_to_be_allocated_on, {"writer"})};
    if (shared) {
        auto reader = test_helpers::MakeMinimalGen1DMKernel("reader", DataMovementProcessor::RISCV_1);
        reader.scratchpad_bindings = spec.kernels[0].scratchpad_bindings;
        spec.kernels.push_back(std::move(reader));
        spec.work_units[0].kernels.push_back(KernelSpecName{"reader"});
        spec.scratchpads[0].advanced_options.allow_instance_multi_binding = true;
    }
    return spec;
}

ProgramSpec MakeScratchpadSpecWithArenaNodesForTest(const NodeRangeSet& nodes) {
    ProgramSpec spec = BorrowedScratchpadSpec();
    spec.advanced_options.sram_arena_parameters[0].properties =
        RangeLockstepSRAMArenaProperties{.banks_to_be_allocated_on = nodes, .per_bank_size = 4096};
    return spec;
}

ProgramRunArgs AllocationArgs(const SRAMArena& allocation, const char* name = "arena") {
    ProgramRunArgs args;
    args.advanced_options.sram_arena_args = {{SRAMArenaParamName{name}, std::cref(allocation)}};
    return args;
}

uint32_t BorrowedAddress(const SRAMArena& allocation) {
    return std::visit(
        [](const auto& layout) { return static_cast<uint32_t>(layout.addr) + 1024; }, allocation.impl().layout);
}

const SRAMArena* AllocationArgument(const ProgramRunArgs& args, const char* name) {
    const auto argument = args.advanced_options.sram_arena_args.get(SRAMArenaParamName{name});
    return argument ? &argument->get() : nullptr;
}

std::array<uint32_t, 2> ScratchpadAddresses(Program& program, const char* kernel_name = "writer") {
    auto kernel = program.impl().get_kernel_by_spec_name(kernel_name);
    const auto& binding = kernel->scratchpad_binding_handles().at(0);
    return {binding.allocated_address, kernel->common_runtime_args_data()[binding.addr_crta_word]};
}

ProgramSpec MakeScratchpadSpecWithTwoArenasForTest() {
    ProgramSpec spec = BorrowedScratchpadSpec();
    ScratchpadSpec second_pad{.unique_id = ScratchpadSpecName{"pad2"}, .size_per_node = 1024};
    second_pad.advanced_options.allocation_override = SRAMArenaAllocation{.arena = SRAMArenaParamName{"second"}};
    spec.scratchpads.push_back(std::move(second_pad));
    spec.kernels[0].scratchpad_bindings.push_back(
        {.scratchpad_spec_name = ScratchpadSpecName{"pad2"}, .accessor_name = "pad2"});
    spec.advanced_options.sram_arena_parameters.push_back(
        {.unique_id = SRAMArenaParamName{"second"}, .properties = AllocationProperties()});
    return spec;
}

PrefetcherPipeSpaceConfig MakePrefetcherPipeSpaceConfigOnFirstScratchpadNodeForTest() {
    return {
        .sender_cores = CoreRangeSet(CoreRange(CoreCoord{0, 0})),
        .receiver_domain = CoreRangeSet(CoreRange(CoreCoord{0, 1})),
        .ring_size = 4096,
        .max_receivers_per_pipe = 1};
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaScratchpadWaitsForRunArgs) {
    // A borrowed scratchpad remains unassigned until its allocation is supplied.
    auto program = MakeProgramFromSpec(*mesh_device_, BorrowedScratchpadSpec());
    EXPECT_THAT(ScratchpadAddresses(program), ::testing::Each(0u));
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaScratchpadMissingArgument) {
    // Full argument setting requires every declared allocation.
    auto program = MakeProgramFromSpec(*mesh_device_, BorrowedScratchpadSpec());
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, {}); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("has no SRAMArenaArgument")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaScratchpadUnknownArgument) {
    // Allocation arguments must name a declared parameter.
    auto arena = SRAMArena::allocate(*mesh_device_, AllocationProperties());
    auto program = MakeProgramFromSpec(*mesh_device_, BorrowedScratchpadSpec());
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, AllocationArgs(arena, "unknown")); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("unknown SRAMArenaParameter")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaScratchpadSizeMismatch) {
    // Allocation size must match the declaration, even when the scratchpad fits.
    auto properties = AllocationProperties();
    properties.per_bank_size *= 2;
    auto arena = SRAMArena::allocate(*mesh_device_, properties);
    auto program = MakeProgramFromSpec(*mesh_device_, BorrowedScratchpadSpec());
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, AllocationArgs(arena)); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("properties do not match")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaScratchpadKindMismatch) {
    // A lockstep allocation cannot satisfy a range-lockstep declaration.
    auto arena = SRAMArena::allocate(*mesh_device_, LockstepSRAMArenaProperties{.per_bank_size = 4096});
    auto program = MakeProgramFromSpec(*mesh_device_, BorrowedScratchpadSpec());
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, AllocationArgs(arena)); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("properties do not match")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaScratchpadNodeMismatch) {
    // The allocation node set must match the declared node set.
    auto properties = AllocationProperties();
    properties.banks_to_be_allocated_on = NodeRangeSet{NodeRange({0, 0}, {2, 0})};
    auto arena = SRAMArena::allocate(*mesh_device_, properties);
    auto program = MakeProgramFromSpec(*mesh_device_, BorrowedScratchpadSpec());
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, AllocationArgs(arena)); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("properties do not match")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaSetDifferentRangeSplits) {
    // Full argument sets accept the same nodes described by different range splits.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    const NodeRangeSet declared_nodes{NodeRange({0, 0}, {1, 0})};
    const NodeRangeSet arena_nodes{std::vector<NodeRange>{NodeRange({0, 0}, {0, 0}), NodeRange({1, 0}, {1, 0})}};
    auto arena = SRAMArena::allocate(fixture_device, RangeLockstepSRAMArenaProperties{arena_nodes, 4096});
    auto program = MakeProgramFromSpec(fixture_device, MakeScratchpadSpecWithArenaNodesForTest(declared_nodes));
    EXPECT_NO_THROW(SetProgramRunArgs(program, AllocationArgs(arena)));
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaUpdateDifferentRangeSplits) {
    // Partial updates accept the same nodes described by different range splits.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    const NodeRangeSet declared_nodes{NodeRange({0, 0}, {1, 0})};
    const NodeRangeSet arena_nodes{std::vector<NodeRange>{NodeRange({0, 0}, {0, 0}), NodeRange({1, 0}, {1, 0})}};
    auto initial = SRAMArena::allocate(fixture_device, RangeLockstepSRAMArenaProperties{declared_nodes, 4096});
    auto arena = SRAMArena::allocate(fixture_device, RangeLockstepSRAMArenaProperties{arena_nodes, 4096});
    auto program = MakeProgramFromSpec(fixture_device, MakeScratchpadSpecWithArenaNodesForTest(declared_nodes));
    SetProgramRunArgs(program, AllocationArgs(initial));
    EXPECT_NO_THROW(UpdateProgramRunArgs(program, AllocationArgs(arena)));
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaSetDifferentRangeOrder) {
    // Full argument sets accept the same node ranges in a different order.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    const NodeRangeSet declared_nodes{std::vector<NodeRange>{NodeRange({0, 0}, {0, 0}), NodeRange({1, 0}, {1, 0})}};
    const NodeRangeSet arena_nodes{std::vector<NodeRange>{NodeRange({1, 0}, {1, 0}), NodeRange({0, 0}, {0, 0})}};
    auto arena = SRAMArena::allocate(fixture_device, RangeLockstepSRAMArenaProperties{arena_nodes, 4096});
    auto program = MakeProgramFromSpec(fixture_device, MakeScratchpadSpecWithArenaNodesForTest(declared_nodes));
    EXPECT_NO_THROW(SetProgramRunArgs(program, AllocationArgs(arena)));
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaUpdateDifferentRangeOrder) {
    // Partial updates accept the same node ranges in a different order.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    const NodeRangeSet declared_nodes{std::vector<NodeRange>{NodeRange({0, 0}, {0, 0}), NodeRange({1, 0}, {1, 0})}};
    const NodeRangeSet arena_nodes{std::vector<NodeRange>{NodeRange({1, 0}, {1, 0}), NodeRange({0, 0}, {0, 0})}};
    auto initial = SRAMArena::allocate(fixture_device, RangeLockstepSRAMArenaProperties{declared_nodes, 4096});
    auto arena = SRAMArena::allocate(fixture_device, RangeLockstepSRAMArenaProperties{arena_nodes, 4096});
    auto program = MakeProgramFromSpec(fixture_device, MakeScratchpadSpecWithArenaNodesForTest(declared_nodes));
    SetProgramRunArgs(program, AllocationArgs(initial));
    EXPECT_NO_THROW(UpdateProgramRunArgs(program, AllocationArgs(arena)));
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaSetDifferentNodesWithSameCount) {
    // Full argument sets reject different node sets with the same node count.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    const NodeRangeSet declared_nodes{NodeRange({0, 0}, {1, 0})};
    const NodeRangeSet arena_nodes{NodeRange({1, 0}, {2, 0})};
    auto arena = SRAMArena::allocate(fixture_device, RangeLockstepSRAMArenaProperties{arena_nodes, 4096});
    auto program = MakeProgramFromSpec(fixture_device, MakeScratchpadSpecWithArenaNodesForTest(declared_nodes));
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, AllocationArgs(arena)); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("properties do not match")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaUpdateDifferentNodesWithSameCount) {
    // Partial updates reject different node sets with the same node count.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    const NodeRangeSet declared_nodes{NodeRange({0, 0}, {1, 0})};
    const NodeRangeSet arena_nodes{NodeRange({1, 0}, {2, 0})};
    auto initial = SRAMArena::allocate(fixture_device, RangeLockstepSRAMArenaProperties{declared_nodes, 4096});
    auto arena = SRAMArena::allocate(fixture_device, RangeLockstepSRAMArenaProperties{arena_nodes, 4096});
    auto program = MakeProgramFromSpec(fixture_device, MakeScratchpadSpecWithArenaNodesForTest(declared_nodes));
    SetProgramRunArgs(program, AllocationArgs(initial));
    EXPECT_THAT(
        [&] { UpdateProgramRunArgs(program, AllocationArgs(arena)); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("properties do not match")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaScratchpadDeviceMismatch) {
    // An allocation from another MeshDevice cannot be bound to the program.
    auto other = mesh_device_->create_submesh(distributed::MeshShape{1, 1});
    auto arena = SRAMArena::allocate(*other, AllocationProperties());
    auto program = MakeProgramFromSpec(*mesh_device_, BorrowedScratchpadSpec());
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, AllocationArgs(arena)); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("different MeshDevice")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaScratchpadSharedBinding) {
    // All kernels sharing a scratchpad receive the same borrowed address.
    auto arena = SRAMArena::allocate(*mesh_device_, AllocationProperties());
    auto program = MakeProgramFromSpec(*mesh_device_, BorrowedScratchpadSpec(true));
    SetProgramRunArgs(program, AllocationArgs(arena));
    EXPECT_THAT(ScratchpadAddresses(program, "writer"), ::testing::Each(BorrowedAddress(arena)));
    EXPECT_THAT(ScratchpadAddresses(program, "reader"), ::testing::Each(BorrowedAddress(arena)));
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaScratchpadSharedRebinding) {
    // A partial allocation update refreshes every shared binding and argument slot.
    auto first = SRAMArena::allocate(*mesh_device_, AllocationProperties());
    auto second = SRAMArena::allocate(*mesh_device_, AllocationProperties());
    auto program = MakeProgramFromSpec(*mesh_device_, BorrowedScratchpadSpec(true));
    SetProgramRunArgs(program, AllocationArgs(first));
    UpdateProgramRunArgs(program, AllocationArgs(second));
    EXPECT_THAT(ScratchpadAddresses(program, "writer"), ::testing::Each(BorrowedAddress(second)));
    EXPECT_THAT(ScratchpadAddresses(program, "reader"), ::testing::Each(BorrowedAddress(second)));
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaScratchpadPreservedDuringReallocation) {
    // Repeating program-local allocation preserves an externally bound scratchpad address.
    auto arena = SRAMArena::allocate(*mesh_device_, AllocationProperties());
    auto program = MakeProgramFromSpec(*mesh_device_, BorrowedScratchpadSpec(true));
    SetProgramRunArgs(program, AllocationArgs(arena));
    program.impl().invalidate_dataflow_buffer_allocation();
    program.impl().compile_and_allocate(mesh_device_.get(), false);
    EXPECT_THAT(ScratchpadAddresses(program, "writer"), ::testing::Each(BorrowedAddress(arena)));
    EXPECT_THAT(ScratchpadAddresses(program, "reader"), ::testing::Each(BorrowedAddress(arena)));
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaScratchpadPartialUnknownArgument) {
    // Partial updates reject allocation names absent from the declaration.
    auto arena = SRAMArena::allocate(*mesh_device_, AllocationProperties());
    auto program = MakeProgramFromSpec(*mesh_device_, BorrowedScratchpadSpec());
    SetProgramRunArgs(program, AllocationArgs(arena));
    EXPECT_THAT(
        [&] { UpdateProgramRunArgs(program, AllocationArgs(arena, "unknown")); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("unknown SRAMArenaParameter")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaScratchpadRejectedUpdatePreservesBinding) {
    // A rejected allocation update leaves the existing address intact.
    auto arena = SRAMArena::allocate(*mesh_device_, AllocationProperties());
    auto invalid = SRAMArena::allocate(*mesh_device_, LockstepSRAMArenaProperties{.per_bank_size = 4096});
    auto program = MakeProgramFromSpec(*mesh_device_, BorrowedScratchpadSpec());
    SetProgramRunArgs(program, AllocationArgs(arena));
    EXPECT_THROW(UpdateProgramRunArgs(program, AllocationArgs(invalid)), std::runtime_error);
    EXPECT_THAT(ScratchpadAddresses(program), ::testing::Each(BorrowedAddress(arena)));
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaMergeDisjointArguments) {
    // Argument merging retains allocations supplied by each input.
    auto first = SRAMArena::allocate(*mesh_device_, AllocationProperties());
    auto second = SRAMArena::allocate(*mesh_device_, AllocationProperties());
    std::vector<ProgramRunArgs> rest{AllocationArgs(second, "second")};
    auto merged = MergeProgramRunArgs(AllocationArgs(first), rest);
    EXPECT_EQ(merged.advanced_options.sram_arena_args.size(), 2u);
    EXPECT_EQ(AllocationArgument(merged, "arena"), &first);
    EXPECT_EQ(AllocationArgument(merged, "second"), &second);
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaMergeDuplicateArguments) {
    // Argument merging rejects an allocation parameter supplied by two inputs.
    auto arena = SRAMArena::allocate(*mesh_device_, AllocationProperties());
    std::vector<ProgramRunArgs> rest{AllocationArgs(arena)};
    EXPECT_THAT(
        [&] { MergeProgramRunArgs(AllocationArgs(arena), rest); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("specified in more than one ProgramRunArgs")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaMergeUncheckedKeepsFirstArgument) {
    // Unchecked merging keeps the first allocation when a name appears twice.
    auto first = SRAMArena::allocate(*mesh_device_, AllocationProperties());
    auto second = SRAMArena::allocate(*mesh_device_, AllocationProperties());
    std::vector<ProgramRunArgs> rest{AllocationArgs(second)};
    auto merged = MergeProgramRunArgs(AllocationArgs(first), rest, /*skip_validation=*/true);
    EXPECT_EQ(AllocationArgument(merged, "arena"), &first);
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaMovedFromValidatedSetFails) {
    // Binding a moved-from arena reports an error, including when validation is skipped.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto arena = SRAMArena::allocate(fixture_device, AllocationProperties());
    auto program = MakeProgramFromSpec(fixture_device, BorrowedScratchpadSpec());
    auto owner = std::move(arena);
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, AllocationArgs(arena), /*skip_validation=*/false); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("SRAMArena is in a moved-from state")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaMovedFromUncheckedSetFails) {
    // Binding a moved-from arena reports an error, including when validation is skipped.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto arena = SRAMArena::allocate(fixture_device, AllocationProperties());
    auto program = MakeProgramFromSpec(fixture_device, BorrowedScratchpadSpec());
    auto owner = std::move(arena);
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, AllocationArgs(arena), /*skip_validation=*/true); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("SRAMArena is in a moved-from state")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaMovedFromValidatedUpdateFails) {
    // Binding a moved-from arena reports an error, including when validation is skipped.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto arena = SRAMArena::allocate(fixture_device, AllocationProperties());
    auto program = MakeProgramFromSpec(fixture_device, BorrowedScratchpadSpec());
    SetProgramRunArgs(program, AllocationArgs(arena));
    auto owner = std::move(arena);
    EXPECT_THAT(
        [&] { UpdateProgramRunArgs(program, AllocationArgs(arena), /*skip_validation=*/false); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("SRAMArena is in a moved-from state")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaMovedFromUncheckedUpdateFails) {
    // Binding a moved-from arena reports an error, including when validation is skipped.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto arena = SRAMArena::allocate(fixture_device, AllocationProperties());
    auto program = MakeProgramFromSpec(fixture_device, BorrowedScratchpadSpec());
    SetProgramRunArgs(program, AllocationArgs(arena));
    auto owner = std::move(arena);
    EXPECT_THAT(
        [&] { UpdateProgramRunArgs(program, AllocationArgs(arena), /*skip_validation=*/true); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("SRAMArena is in a moved-from state")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaMovedOwnerCanBind) {
    // The new owner of a moved reservation can supply program arena storage.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto arena = SRAMArena::allocate(fixture_device, AllocationProperties());
    auto owner = std::move(arena);
    auto program = MakeProgramFromSpec(fixture_device, BorrowedScratchpadSpec());
    SetProgramRunArgs(program, AllocationArgs(owner));
    EXPECT_THAT(ScratchpadAddresses(program), ::testing::Each(BorrowedAddress(owner)));
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaSetAcceptsEveryDeclaredArena) {
    // A program that declares two arenas accepts a full argument set that supplies both.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto first = SRAMArena::allocate(fixture_device, AllocationProperties());
    auto second = SRAMArena::allocate(fixture_device, AllocationProperties());
    auto program = MakeProgramFromSpec(fixture_device, MakeScratchpadSpecWithTwoArenasForTest());
    ProgramRunArgs args;
    args.advanced_options.sram_arena_args = {
        {SRAMArenaParamName{"arena"}, std::cref(first)}, {SRAMArenaParamName{"second"}, std::cref(second)}};
    EXPECT_NO_THROW(SetProgramRunArgs(program, args));
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaScratchpadLeavesNodesOpenForPersistentL1) {
    // A Program whose only scratchpad lives in an SRAMArena does not block a later PrefetcherPipeSpace on its nodes.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    Program program = MakeProgramFromSpec(fixture_device, BorrowedScratchpadSpec());
    const PrefetcherPipeSpaceConfig space_config = MakePrefetcherPipeSpaceConfigOnFirstScratchpadNodeForTest();
    EXPECT_NO_THROW({ PrefetcherPipeSpace space = CreatePrefetcherPipeSpace(fixture_device, space_config); });
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaSharedScratchpadLeavesNodesOpenForPersistentL1) {
    // A Program whose only scratchpad is a shared SRAMArena scratchpad does not block a later PrefetcherPipeSpace.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    Program program = MakeProgramFromSpec(fixture_device, BorrowedScratchpadSpec(/*shared=*/true));
    const PrefetcherPipeSpaceConfig space_config = MakePrefetcherPipeSpaceConfigOnFirstScratchpadNodeForTest();
    EXPECT_NO_THROW({ PrefetcherPipeSpace space = CreatePrefetcherPipeSpace(fixture_device, space_config); });
}

TEST_F(ProgramRunArgsTestGen1, CPU_SRAMArenaScratchpadRangeArenaForLockstepDeclaration) {
    // A range-lockstep allocation cannot satisfy a lockstep declaration.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    ProgramSpec spec = BorrowedScratchpadSpec();
    spec.advanced_options.sram_arena_parameters[0].properties = LockstepSRAMArenaProperties{.per_bank_size = 4096};
    auto arena = SRAMArena::allocate(fixture_device, AllocationProperties());
    auto program = MakeProgramFromSpec(fixture_device, spec);
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, AllocationArgs(arena)); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("properties do not match")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
