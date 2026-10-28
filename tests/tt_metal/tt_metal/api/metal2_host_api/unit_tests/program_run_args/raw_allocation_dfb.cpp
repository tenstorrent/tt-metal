// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <variant>

#include <tt-metalium/experimental/allocation/raw_allocation.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "impl/allocation/raw_allocation_impl.hpp"
#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

class RawAllocationDFBTest : public test_helpers::ProgramRunArgsTestGen1 {
protected:
    distributed::MeshDevice& fixture_device() { return *mesh_device_; }
};

RangeLockstepAllocationProperties ArenaProperties(uint32_t per_bank_size) {
    return {.banks_to_be_allocated_on = NodeRangeSet{NodeRange({0, 0}, {1, 0})}, .per_bank_size = per_bank_size};
}

ProgramSpec BorrowedDFBSpec(uint32_t entry_size, uint32_t num_entries, uint32_t offset) {
    ProgramSpec spec;
    spec.kernels = {
        test_helpers::MakeMinimalGen1DMKernel("producer", DataMovementProcessor::RISCV_0),
        test_helpers::MakeMinimalGen1DMKernel("consumer", DataMovementProcessor::RISCV_1)};
    spec.kernels[0].dfb_bindings = {ProducerOf(DFBSpecName{"fifo"}, "fifo")};
    spec.kernels[1].dfb_bindings = {ConsumerOf(DFBSpecName{"fifo"}, "fifo")};
    spec.dataflow_buffers = {test_helpers::MakeMinimalDFB("fifo", entry_size, num_entries)};
    spec.dataflow_buffers[0].borrowed_from = RawAllocationParamName{"arena"};
    spec.dataflow_buffers[0].advanced_options.borrowed_memory_offset = offset;
    spec.advanced_options.raw_allocation_parameters = {
        {.unique_id = RawAllocationParamName{"arena"}, .properties = ArenaProperties(4096)}};
    spec.work_units = {test_helpers::MakeMinimalWorkUnit(
        "work", ArenaProperties(4096).banks_to_be_allocated_on, {"producer", "consumer"})};
    return spec;
}

uint32_t ArenaBase(const RawAllocation& arena) {
    return std::visit([](const auto& layout) { return static_cast<uint32_t>(layout.addr); }, arena.impl().layout);
}

dfb::detail::DataflowBufferImpl& BorrowedDFB(Program& program) {
    return *program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("fifo"));
}

TEST_F(RawAllocationDFBTest, WaitsForAllocationArgument) {
    // A borrowed DFB has no storage address before its allocation argument is supplied.
    ProgramSpec spec = BorrowedDFBSpec(/*entry_size=*/512, /*num_entries=*/4, /*offset=*/1024);
    Program program = MakeProgramFromSpec(fixture_device(), spec);
    EXPECT_TRUE(BorrowedDFB(program).borrows_memory());
    EXPECT_EQ(BorrowedDFB(program).borrowed_addr_, 0u);
}

TEST_F(RawAllocationDFBTest, BindsAtDeclaredOffset) {
    // Supplying an allocation places the DFB at its exact base plus the declared offset.
    RawAllocation arena = allocate_on_sram(fixture_device(), ArenaProperties(4096));
    ProgramSpec spec = BorrowedDFBSpec(/*entry_size=*/512, /*num_entries=*/4, /*offset=*/1024);
    Program program = MakeProgramFromSpec(fixture_device(), spec);
    ProgramRunArgs args;
    args.advanced_options.raw_allocation_args = {{RawAllocationParamName{"arena"}, std::cref(arena)}};
    SetProgramRunArgs(program, args);
    EXPECT_EQ(BorrowedDFB(program).borrowed_addr_, ArenaBase(arena) + 1024);
}

TEST_F(RawAllocationDFBTest, FullResetRebindsEveryNode) {
    // A full argument reset changes the DFB address on every occupied node.
    RawAllocation first = allocate_on_sram(fixture_device(), ArenaProperties(4096));
    RawAllocation second = allocate_on_sram(fixture_device(), ArenaProperties(4096));
    ProgramSpec spec = BorrowedDFBSpec(/*entry_size=*/512, /*num_entries=*/4, /*offset=*/1024);
    Program program = MakeProgramFromSpec(fixture_device(), spec);
    ProgramRunArgs args;
    args.advanced_options.raw_allocation_args = {{RawAllocationParamName{"arena"}, std::cref(first)}};
    SetProgramRunArgs(program, args);
    args.advanced_options.raw_allocation_args = {{RawAllocationParamName{"arena"}, std::cref(second)}};
    SetProgramRunArgs(program, args);
    EXPECT_EQ(BorrowedDFB(program).core_lookup_.at(NodeCoord{0, 0}).second, ArenaBase(second) + 1024);
    EXPECT_EQ(BorrowedDFB(program).core_lookup_.at(NodeCoord{1, 0}).second, ArenaBase(second) + 1024);
}

TEST_F(RawAllocationDFBTest, PartialUpdateRebindsEveryNode) {
    // An allocation-only update changes the DFB address on every occupied node.
    RawAllocation first = allocate_on_sram(fixture_device(), ArenaProperties(4096));
    RawAllocation second = allocate_on_sram(fixture_device(), ArenaProperties(4096));
    ProgramSpec spec = BorrowedDFBSpec(/*entry_size=*/512, /*num_entries=*/4, /*offset=*/1024);
    Program program = MakeProgramFromSpec(fixture_device(), spec);
    ProgramRunArgs args;
    args.advanced_options.raw_allocation_args = {{RawAllocationParamName{"arena"}, std::cref(first)}};
    SetProgramRunArgs(program, args);
    args.advanced_options.raw_allocation_args = {{RawAllocationParamName{"arena"}, std::cref(second)}};
    UpdateProgramRunArgs(program, args);
    EXPECT_EQ(BorrowedDFB(program).core_lookup_.at(NodeCoord{0, 0}).second, ArenaBase(second) + 1024);
    EXPECT_EQ(BorrowedDFB(program).core_lookup_.at(NodeCoord{1, 0}).second, ArenaBase(second) + 1024);
}

TEST_F(RawAllocationDFBTest, PartialUpdateRetainsOmittedAllocation) {
    // An update without an allocation argument preserves the DFB address.
    RawAllocation arena = allocate_on_sram(fixture_device(), ArenaProperties(4096));
    ProgramSpec spec = BorrowedDFBSpec(/*entry_size=*/512, /*num_entries=*/4, /*offset=*/1024);
    Program program = MakeProgramFromSpec(fixture_device(), spec);
    ProgramRunArgs args;
    args.advanced_options.raw_allocation_args = {{RawAllocationParamName{"arena"}, std::cref(arena)}};
    SetProgramRunArgs(program, args);
    UpdateProgramRunArgs(program, {});
    EXPECT_EQ(BorrowedDFB(program).borrowed_addr_, ArenaBase(arena) + 1024);
}

TEST_F(RawAllocationDFBTest, LocalReallocationPreservesBorrowedAddress) {
    // Repeating program-local allocation preserves the borrowed DFB address on every node.
    RawAllocation arena = allocate_on_sram(fixture_device(), ArenaProperties(4096));
    ProgramSpec spec = BorrowedDFBSpec(/*entry_size=*/512, /*num_entries=*/4, /*offset=*/1024);
    Program program = MakeProgramFromSpec(fixture_device(), spec);
    ProgramRunArgs args;
    args.advanced_options.raw_allocation_args = {{RawAllocationParamName{"arena"}, std::cref(arena)}};
    SetProgramRunArgs(program, args);
    program.impl().compile_and_allocate(&fixture_device(), false);
    program.impl().invalidate_dataflow_buffer_allocation();
    program.impl().compile_and_allocate(&fixture_device(), false);
    EXPECT_EQ(BorrowedDFB(program).core_lookup_.at(NodeCoord{0, 0}).second, ArenaBase(arena) + 1024);
    EXPECT_EQ(BorrowedDFB(program).core_lookup_.at(NodeCoord{1, 0}).second, ArenaBase(arena) + 1024);
}

TEST_F(RawAllocationDFBTest, FullResetRequiresAllocationArgument) {
    // A full reset must supply the allocation even when the DFB already has an address.
    RawAllocation arena = allocate_on_sram(fixture_device(), ArenaProperties(4096));
    ProgramSpec spec = BorrowedDFBSpec(/*entry_size=*/512, /*num_entries=*/4, /*offset=*/1024);
    Program program = MakeProgramFromSpec(fixture_device(), spec);
    ProgramRunArgs args;
    args.advanced_options.raw_allocation_args = {{RawAllocationParamName{"arena"}, std::cref(arena)}};
    SetProgramRunArgs(program, args);
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, {}); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("has no RawAllocationArgument")));
}

TEST_F(RawAllocationDFBTest, FullResizeFitsExactlyAfterOffset) {
    // A full argument set accepts a DFB whose offset and resized storage exactly fill the reservation.
    RawAllocation arena = allocate_on_sram(fixture_device(), ArenaProperties(4096));
    ProgramSpec spec = BorrowedDFBSpec(/*entry_size=*/512, /*num_entries=*/4, /*offset=*/1024);
    Program program = MakeProgramFromSpec(fixture_device(), spec);
    ProgramRunArgs args;
    args.advanced_options.raw_allocation_args = {{RawAllocationParamName{"arena"}, std::cref(arena)}};
    args.dfb_run_overrides = {{.dfb = DFBSpecName{"fifo"}, .num_entries = 6}};
    SetProgramRunArgs(program, args);
    EXPECT_EQ(BorrowedDFB(program).total_size(), 3072u);
    EXPECT_EQ(BorrowedDFB(program).borrowed_addr_, ArenaBase(arena) + 1024);
}

TEST_F(RawAllocationDFBTest, PartialResizeDoesNotRequireAllocationArgument) {
    // A partial resize can fill the remaining reservation without supplying the allocation again.
    RawAllocation arena = allocate_on_sram(fixture_device(), ArenaProperties(4096));
    ProgramSpec spec = BorrowedDFBSpec(/*entry_size=*/512, /*num_entries=*/4, /*offset=*/1024);
    Program program = MakeProgramFromSpec(fixture_device(), spec);
    ProgramRunArgs args;
    args.advanced_options.raw_allocation_args = {{RawAllocationParamName{"arena"}, std::cref(arena)}};
    SetProgramRunArgs(program, args);
    ProgramRunArgs update;
    update.dfb_run_overrides = {{.dfb = DFBSpecName{"fifo"}, .num_entries = 6}};
    UpdateProgramRunArgs(program, update);
    EXPECT_EQ(BorrowedDFB(program).total_size(), 3072u);
    EXPECT_EQ(BorrowedDFB(program).borrowed_addr_, ArenaBase(arena) + 1024);
}

TEST_F(RawAllocationDFBTest, FullResizeRejectsStorageBeyondOffsetLimit) {
    // A DFB that fits only when its offset is ignored must be rejected by a full argument set.
    RawAllocation arena = allocate_on_sram(fixture_device(), ArenaProperties(4096));
    ProgramSpec spec = BorrowedDFBSpec(/*entry_size=*/512, /*num_entries=*/4, /*offset=*/1024);
    Program program = MakeProgramFromSpec(fixture_device(), spec);
    ProgramRunArgs args;
    args.advanced_options.raw_allocation_args = {{RawAllocationParamName{"arena"}, std::cref(arena)}};
    args.dfb_run_overrides = {{.dfb = DFBSpecName{"fifo"}, .num_entries = 8}};
    EXPECT_THROW(SetProgramRunArgs(program, args), std::runtime_error);
}

TEST_F(RawAllocationDFBTest, PartialResizeRejectsStorageBeyondOffsetLimit) {
    // Omitting the allocation does not bypass the offset-inclusive resize bound.
    RawAllocation arena = allocate_on_sram(fixture_device(), ArenaProperties(4096));
    ProgramSpec spec = BorrowedDFBSpec(/*entry_size=*/512, /*num_entries=*/4, /*offset=*/1024);
    Program program = MakeProgramFromSpec(fixture_device(), spec);
    ProgramRunArgs args;
    args.advanced_options.raw_allocation_args = {{RawAllocationParamName{"arena"}, std::cref(arena)}};
    SetProgramRunArgs(program, args);
    ProgramRunArgs update;
    update.dfb_run_overrides = {{.dfb = DFBSpecName{"fifo"}, .num_entries = 8}};
    EXPECT_THROW(UpdateProgramRunArgs(program, update), std::runtime_error);
}

TEST_F(RawAllocationDFBTest, EntrySizeOverrideChecksReservationBound) {
    // An entry-size override must fit with the current entry count and borrowing offset.
    RawAllocation arena = allocate_on_sram(fixture_device(), ArenaProperties(4096));
    ProgramSpec spec = BorrowedDFBSpec(/*entry_size=*/512, /*num_entries=*/4, /*offset=*/1024);
    Program program = MakeProgramFromSpec(fixture_device(), spec);
    ProgramRunArgs args;
    args.advanced_options.raw_allocation_args = {{RawAllocationParamName{"arena"}, std::cref(arena)}};
    SetProgramRunArgs(program, args);
    ProgramRunArgs update;
    update.dfb_run_overrides = {{.dfb = DFBSpecName{"fifo"}, .entry_size = 1024}};
    EXPECT_THROW(UpdateProgramRunArgs(program, update), std::runtime_error);
}

TEST_F(RawAllocationDFBTest, CombinedSizeOverridesUseResultingSize) {
    // Entry size and count overrides are checked together against the available storage.
    RawAllocation arena = allocate_on_sram(fixture_device(), ArenaProperties(4096));
    ProgramSpec spec = BorrowedDFBSpec(/*entry_size=*/512, /*num_entries=*/4, /*offset=*/1024);
    Program program = MakeProgramFromSpec(fixture_device(), spec);
    ProgramRunArgs args;
    args.advanced_options.raw_allocation_args = {{RawAllocationParamName{"arena"}, std::cref(arena)}};
    args.dfb_run_overrides = {{.dfb = DFBSpecName{"fifo"}, .entry_size = 1536, .num_entries = 2}};
    SetProgramRunArgs(program, args);
    EXPECT_EQ(BorrowedDFB(program).config.entry_size, 1536u);
    EXPECT_EQ(BorrowedDFB(program).config.num_entries, 2u);
}

TEST_F(RawAllocationDFBTest, LaterResizeUsesCurrentEntryCount) {
    // An omitted entry count retains its previous override during a later resize.
    RawAllocation arena = allocate_on_sram(fixture_device(), ArenaProperties(4096));
    ProgramSpec spec = BorrowedDFBSpec(/*entry_size=*/512, /*num_entries=*/4, /*offset=*/1024);
    Program program = MakeProgramFromSpec(fixture_device(), spec);
    ProgramRunArgs args;
    args.advanced_options.raw_allocation_args = {{RawAllocationParamName{"arena"}, std::cref(arena)}};
    args.dfb_run_overrides = {{.dfb = DFBSpecName{"fifo"}, .num_entries = 2}};
    SetProgramRunArgs(program, args);
    ProgramRunArgs update;
    update.dfb_run_overrides = {{.dfb = DFBSpecName{"fifo"}, .entry_size = 1536}};
    UpdateProgramRunArgs(program, update);
    EXPECT_EQ(BorrowedDFB(program).config.entry_size, 1536u);
    EXPECT_EQ(BorrowedDFB(program).config.num_entries, 2u);
}

}  // namespace
}  // namespace tt::tt_metal::experimental
