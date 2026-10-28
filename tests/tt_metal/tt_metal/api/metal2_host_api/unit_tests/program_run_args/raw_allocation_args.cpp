// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <array>
#include <stdexcept>
#include <vector>

#include <tt-metalium/experimental/allocation/raw_allocation.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "impl/allocation/raw_allocation_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::ProgramRunArgsTestGen1;

RangeLockstepAllocationProperties AllocationProperties() {
    return {.banks_to_be_allocated_on = NodeRangeSet{NodeRange({0, 0}, {1, 0})}, .per_bank_size = 4096};
}

ProgramSpec BorrowedScratchpadSpec(bool shared = false) {
    ProgramSpec spec;
    spec.kernels = {test_helpers::MakeMinimalGen1DMKernel("writer")};
    spec.kernels[0].scratchpad_bindings = {{.scratchpad_spec_name = ScratchpadSpecName{"pad"}, .accessor_name = "pad"}};
    spec.scratchpads = {{.unique_id = ScratchpadSpecName{"pad"}, .size_per_node = 1024}};
    spec.scratchpads[0].advanced_options.borrowed_from = ScratchpadAdvancedOptions::RawAllocationBorrowing{
        .allocation = RawAllocationParamName{"arena"}, .offset = 1024};
    spec.advanced_options.raw_allocation_parameters = {
        {.unique_id = RawAllocationParamName{"arena"}, .properties = AllocationProperties()}};
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

ProgramRunArgs AllocationArgs(const RawAllocation& allocation, const char* name = "arena") {
    ProgramRunArgs args;
    args.advanced_options.raw_allocation_args = {{RawAllocationParamName{name}, std::cref(allocation)}};
    return args;
}

uint32_t BorrowedAddress(const RawAllocation& allocation) {
    return std::visit(
        [](const auto& layout) { return static_cast<uint32_t>(layout.addr) + 1024; }, allocation.impl().layout);
}

const RawAllocation* AllocationArgument(const ProgramRunArgs& args, const char* name) {
    const auto argument = args.advanced_options.raw_allocation_args.get(RawAllocationParamName{name});
    return argument ? &argument->get() : nullptr;
}

std::array<uint32_t, 2> ScratchpadAddresses(Program& program, const char* kernel_name = "writer") {
    auto kernel = program.impl().get_kernel_by_spec_name(kernel_name);
    const auto& binding = kernel->scratchpad_binding_handles().at(0);
    return {binding.allocated_address, kernel->common_runtime_args_data()[binding.addr_crta_word]};
}

TEST_F(ProgramRunArgsTestGen1, CPU_RawAllocationScratchpadWaitsForRunArgs) {
    // A borrowed scratchpad remains unassigned until its allocation is supplied.
    auto program = MakeProgramFromSpec(*mesh_device_, BorrowedScratchpadSpec());
    EXPECT_THAT(ScratchpadAddresses(program), ::testing::Each(0u));
}

TEST_F(ProgramRunArgsTestGen1, CPU_RawAllocationScratchpadMissingArgument) {
    // Full argument setting requires every declared allocation.
    auto program = MakeProgramFromSpec(*mesh_device_, BorrowedScratchpadSpec());
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, {}); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("has no RawAllocationArgument")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_RawAllocationScratchpadUnknownArgument) {
    // Allocation arguments must name a declared parameter.
    auto arena = allocate_on_sram(*mesh_device_, AllocationProperties());
    auto program = MakeProgramFromSpec(*mesh_device_, BorrowedScratchpadSpec());
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, AllocationArgs(arena, "unknown")); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("unknown RawAllocationParameter")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_RawAllocationScratchpadSizeMismatch) {
    // Allocation size must match the declaration, even when the scratchpad fits.
    auto properties = AllocationProperties();
    properties.per_bank_size *= 2;
    auto arena = allocate_on_sram(*mesh_device_, properties);
    auto program = MakeProgramFromSpec(*mesh_device_, BorrowedScratchpadSpec());
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, AllocationArgs(arena)); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("properties do not match")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_RawAllocationScratchpadKindMismatch) {
    // A lockstep allocation cannot satisfy a range-lockstep declaration.
    auto arena = allocate_on_sram(*mesh_device_, LockstepAllocationProperties{.per_bank_size = 4096});
    auto program = MakeProgramFromSpec(*mesh_device_, BorrowedScratchpadSpec());
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, AllocationArgs(arena)); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("properties do not match")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_RawAllocationScratchpadNodeMismatch) {
    // The allocation node set must match the declared node set.
    auto properties = AllocationProperties();
    properties.banks_to_be_allocated_on = NodeRangeSet{NodeRange({0, 0}, {2, 0})};
    auto arena = allocate_on_sram(*mesh_device_, properties);
    auto program = MakeProgramFromSpec(*mesh_device_, BorrowedScratchpadSpec());
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, AllocationArgs(arena)); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("properties do not match")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_RawAllocationScratchpadDeviceMismatch) {
    // An allocation from another MeshDevice cannot be bound to the program.
    auto other = mesh_device_->create_submesh(distributed::MeshShape{1, 1});
    auto arena = allocate_on_sram(*other, AllocationProperties());
    auto program = MakeProgramFromSpec(*mesh_device_, BorrowedScratchpadSpec());
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, AllocationArgs(arena)); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("different MeshDevice")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_RawAllocationScratchpadSharedBinding) {
    // All kernels sharing a scratchpad receive the same borrowed address.
    auto arena = allocate_on_sram(*mesh_device_, AllocationProperties());
    auto program = MakeProgramFromSpec(*mesh_device_, BorrowedScratchpadSpec(true));
    SetProgramRunArgs(program, AllocationArgs(arena));
    EXPECT_THAT(ScratchpadAddresses(program, "writer"), ::testing::Each(BorrowedAddress(arena)));
    EXPECT_THAT(ScratchpadAddresses(program, "reader"), ::testing::Each(BorrowedAddress(arena)));
}

TEST_F(ProgramRunArgsTestGen1, CPU_RawAllocationScratchpadSharedRebinding) {
    // A partial allocation update refreshes every shared binding and argument slot.
    auto first = allocate_on_sram(*mesh_device_, AllocationProperties());
    auto second = allocate_on_sram(*mesh_device_, AllocationProperties());
    auto program = MakeProgramFromSpec(*mesh_device_, BorrowedScratchpadSpec(true));
    SetProgramRunArgs(program, AllocationArgs(first));
    UpdateProgramRunArgs(program, AllocationArgs(second));
    EXPECT_THAT(ScratchpadAddresses(program, "writer"), ::testing::Each(BorrowedAddress(second)));
    EXPECT_THAT(ScratchpadAddresses(program, "reader"), ::testing::Each(BorrowedAddress(second)));
}

TEST_F(ProgramRunArgsTestGen1, CPU_RawAllocationScratchpadPreservedDuringReallocation) {
    // Repeating program-local allocation preserves an externally bound scratchpad address.
    auto arena = allocate_on_sram(*mesh_device_, AllocationProperties());
    auto program = MakeProgramFromSpec(*mesh_device_, BorrowedScratchpadSpec(true));
    SetProgramRunArgs(program, AllocationArgs(arena));
    program.impl().invalidate_dataflow_buffer_allocation();
    program.impl().compile_and_allocate(mesh_device_.get(), false);
    EXPECT_THAT(ScratchpadAddresses(program, "writer"), ::testing::Each(BorrowedAddress(arena)));
    EXPECT_THAT(ScratchpadAddresses(program, "reader"), ::testing::Each(BorrowedAddress(arena)));
}

TEST_F(ProgramRunArgsTestGen1, CPU_RawAllocationScratchpadPartialUnknownArgument) {
    // Partial updates reject allocation names absent from the declaration.
    auto arena = allocate_on_sram(*mesh_device_, AllocationProperties());
    auto program = MakeProgramFromSpec(*mesh_device_, BorrowedScratchpadSpec());
    SetProgramRunArgs(program, AllocationArgs(arena));
    EXPECT_THAT(
        [&] { UpdateProgramRunArgs(program, AllocationArgs(arena, "unknown")); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("unknown RawAllocationParameter")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_RawAllocationScratchpadRejectedUpdatePreservesBinding) {
    // A rejected allocation update leaves the existing address intact.
    auto arena = allocate_on_sram(*mesh_device_, AllocationProperties());
    auto invalid = allocate_on_sram(*mesh_device_, LockstepAllocationProperties{.per_bank_size = 4096});
    auto program = MakeProgramFromSpec(*mesh_device_, BorrowedScratchpadSpec());
    SetProgramRunArgs(program, AllocationArgs(arena));
    EXPECT_THROW(UpdateProgramRunArgs(program, AllocationArgs(invalid)), std::runtime_error);
    EXPECT_THAT(ScratchpadAddresses(program), ::testing::Each(BorrowedAddress(arena)));
}

TEST_F(ProgramRunArgsTestGen1, CPU_RawAllocationMergeDisjointArguments) {
    // Argument merging retains allocations supplied by each input.
    auto first = allocate_on_sram(*mesh_device_, AllocationProperties());
    auto second = allocate_on_sram(*mesh_device_, AllocationProperties());
    std::vector<ProgramRunArgs> rest{AllocationArgs(second, "second")};
    auto merged = MergeProgramRunArgs(AllocationArgs(first), rest);
    EXPECT_EQ(merged.advanced_options.raw_allocation_args.size(), 2u);
    EXPECT_EQ(AllocationArgument(merged, "arena"), &first);
    EXPECT_EQ(AllocationArgument(merged, "second"), &second);
}

TEST_F(ProgramRunArgsTestGen1, CPU_RawAllocationMergeDuplicateArguments) {
    // Argument merging rejects an allocation parameter supplied by two inputs.
    auto arena = allocate_on_sram(*mesh_device_, AllocationProperties());
    std::vector<ProgramRunArgs> rest{AllocationArgs(arena)};
    EXPECT_THAT(
        [&] { MergeProgramRunArgs(AllocationArgs(arena), rest); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("specified in more than one ProgramRunArgs")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_RawAllocationMergeUncheckedKeepsFirstArgument) {
    // Unchecked merging keeps the first allocation when a name appears twice.
    auto first = allocate_on_sram(*mesh_device_, AllocationProperties());
    auto second = allocate_on_sram(*mesh_device_, AllocationProperties());
    std::vector<ProgramRunArgs> rest{AllocationArgs(second)};
    auto merged = MergeProgramRunArgs(AllocationArgs(first), rest, /*skip_validation=*/true);
    EXPECT_EQ(AllocationArgument(merged, "arena"), &first);
}

}  // namespace
}  // namespace tt::tt_metal::experimental
