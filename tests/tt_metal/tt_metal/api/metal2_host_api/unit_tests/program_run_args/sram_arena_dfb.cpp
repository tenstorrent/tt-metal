// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <variant>

#include <tt-metalium/experimental/sram_arena/sram_arena.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "impl/sram_arena/sram_arena_impl.hpp"
#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

class SRAMArenaDFBTest : public test_helpers::ProgramRunArgsTestGen1 {
protected:
    distributed::MeshDevice& fixture_device() { return *mesh_device_; }
};

RangeLockstepSRAMArenaProperties ArenaProperties(uint32_t per_bank_size) {
    return {.banks_to_be_allocated_on = NodeRangeSet{NodeRange({0, 0}, {1, 0})}, .per_bank_size = per_bank_size};
}

ProgramSpec ArenaDFBSpec(uint32_t entry_size, uint32_t num_entries, uint32_t offset) {
    ProgramSpec spec;
    spec.kernels = {
        test_helpers::MakeMinimalGen1DMKernel("producer", DataMovementProcessor::RISCV_0),
        test_helpers::MakeMinimalGen1DMKernel("consumer", DataMovementProcessor::RISCV_1)};
    spec.kernels[0].dfb_bindings = {ProducerOf(DFBSpecName{"fifo"}, "fifo")};
    spec.kernels[1].dfb_bindings = {ConsumerOf(DFBSpecName{"fifo"}, "fifo")};
    spec.dataflow_buffers = {test_helpers::MakeMinimalDFB("fifo", entry_size, num_entries)};
    spec.dataflow_buffers[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"arena"}, .offset = offset};
    spec.advanced_options.sram_arena_parameters = {
        {.unique_id = SRAMArenaParamName{"arena"}, .properties = ArenaProperties(4096)}};
    spec.work_units = {test_helpers::MakeMinimalWorkUnit(
        "work", ArenaProperties(4096).banks_to_be_allocated_on, {"producer", "consumer"})};
    return spec;
}

uint32_t ArenaBase(const SRAMArena& arena) {
    return std::visit([](const auto& layout) { return static_cast<uint32_t>(layout.addr); }, arena.impl().layout);
}

dfb::detail::DataflowBufferImpl& ArenaDFB(Program& program) {
    return *program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("fifo"));
}

TEST_F(SRAMArenaDFBTest, CPU_WaitsForAllocationArgument) {
    // An arena-allocated DFB has no storage address before its allocation argument is supplied.
    ProgramSpec spec = ArenaDFBSpec(/*entry_size=*/512, /*num_entries=*/4, /*offset=*/1024);
    Program program = MakeProgramFromSpec(fixture_device(), spec);
    EXPECT_TRUE(ArenaDFB(program).borrows_memory());
    EXPECT_EQ(ArenaDFB(program).borrowed_addr_, 0u);
}

TEST_F(SRAMArenaDFBTest, CPU_BindsAtDeclaredOffset) {
    // Supplying an allocation places the DFB at its exact base plus the declared offset.
    SRAMArena arena = SRAMArena::allocate(fixture_device(), ArenaProperties(4096));
    ProgramSpec spec = ArenaDFBSpec(/*entry_size=*/512, /*num_entries=*/4, /*offset=*/1024);
    Program program = MakeProgramFromSpec(fixture_device(), spec);
    ProgramRunArgs args;
    args.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(arena)}};
    SetProgramRunArgs(program, args);
    EXPECT_EQ(ArenaDFB(program).borrowed_addr_, ArenaBase(arena) + 1024);
}

TEST_F(SRAMArenaDFBTest, CPU_FullResetRebindsEveryNode) {
    // A full argument reset changes the DFB address on every occupied node.
    SRAMArena first = SRAMArena::allocate(fixture_device(), ArenaProperties(4096));
    SRAMArena second = SRAMArena::allocate(fixture_device(), ArenaProperties(4096));
    ProgramSpec spec = ArenaDFBSpec(/*entry_size=*/512, /*num_entries=*/4, /*offset=*/1024);
    Program program = MakeProgramFromSpec(fixture_device(), spec);
    ProgramRunArgs args;
    args.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(first)}};
    SetProgramRunArgs(program, args);
    args.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(second)}};
    SetProgramRunArgs(program, args);
    EXPECT_EQ(ArenaDFB(program).core_lookup_.at(NodeCoord{0, 0}).second, ArenaBase(second) + 1024);
    EXPECT_EQ(ArenaDFB(program).core_lookup_.at(NodeCoord{1, 0}).second, ArenaBase(second) + 1024);
}

TEST_F(SRAMArenaDFBTest, CPU_PartialUpdateRebindsEveryNode) {
    // An allocation-only update changes the DFB address on every occupied node.
    SRAMArena first = SRAMArena::allocate(fixture_device(), ArenaProperties(4096));
    SRAMArena second = SRAMArena::allocate(fixture_device(), ArenaProperties(4096));
    ProgramSpec spec = ArenaDFBSpec(/*entry_size=*/512, /*num_entries=*/4, /*offset=*/1024);
    Program program = MakeProgramFromSpec(fixture_device(), spec);
    ProgramRunArgs args;
    args.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(first)}};
    SetProgramRunArgs(program, args);
    args.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(second)}};
    UpdateProgramRunArgs(program, args);
    EXPECT_EQ(ArenaDFB(program).core_lookup_.at(NodeCoord{0, 0}).second, ArenaBase(second) + 1024);
    EXPECT_EQ(ArenaDFB(program).core_lookup_.at(NodeCoord{1, 0}).second, ArenaBase(second) + 1024);
}

TEST_F(SRAMArenaDFBTest, CPU_PartialUpdateRetainsOmittedAllocation) {
    // An update without an allocation argument preserves the DFB address.
    SRAMArena arena = SRAMArena::allocate(fixture_device(), ArenaProperties(4096));
    ProgramSpec spec = ArenaDFBSpec(/*entry_size=*/512, /*num_entries=*/4, /*offset=*/1024);
    Program program = MakeProgramFromSpec(fixture_device(), spec);
    ProgramRunArgs args;
    args.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(arena)}};
    SetProgramRunArgs(program, args);
    UpdateProgramRunArgs(program, {});
    EXPECT_EQ(ArenaDFB(program).borrowed_addr_, ArenaBase(arena) + 1024);
}

TEST_F(SRAMArenaDFBTest, CPU_LocalReallocationPreservesBorrowedAddress) {
    // Repeating program-local allocation preserves the arena-allocated DFB address on every node.
    SRAMArena arena = SRAMArena::allocate(fixture_device(), ArenaProperties(4096));
    ProgramSpec spec = ArenaDFBSpec(/*entry_size=*/512, /*num_entries=*/4, /*offset=*/1024);
    Program program = MakeProgramFromSpec(fixture_device(), spec);
    ProgramRunArgs args;
    args.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(arena)}};
    SetProgramRunArgs(program, args);
    program.impl().compile_and_allocate(&fixture_device(), false);
    program.impl().invalidate_dataflow_buffer_allocation();
    program.impl().compile_and_allocate(&fixture_device(), false);
    EXPECT_EQ(ArenaDFB(program).core_lookup_.at(NodeCoord{0, 0}).second, ArenaBase(arena) + 1024);
    EXPECT_EQ(ArenaDFB(program).core_lookup_.at(NodeCoord{1, 0}).second, ArenaBase(arena) + 1024);
}

TEST_F(SRAMArenaDFBTest, CPU_FullResetRequiresAllocationArgument) {
    // A full reset must supply the allocation even when the DFB already has an address.
    SRAMArena arena = SRAMArena::allocate(fixture_device(), ArenaProperties(4096));
    ProgramSpec spec = ArenaDFBSpec(/*entry_size=*/512, /*num_entries=*/4, /*offset=*/1024);
    Program program = MakeProgramFromSpec(fixture_device(), spec);
    ProgramRunArgs args;
    args.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(arena)}};
    SetProgramRunArgs(program, args);
    EXPECT_THAT(
        [&] { SetProgramRunArgs(program, {}); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("has no SRAMArenaArgument")));
}

TEST_F(SRAMArenaDFBTest, CPU_FullResizeFitsExactlyAfterOffset) {
    // A full argument set accepts a DFB whose offset and resized storage exactly fill the reservation.
    SRAMArena arena = SRAMArena::allocate(fixture_device(), ArenaProperties(4096));
    ProgramSpec spec = ArenaDFBSpec(/*entry_size=*/512, /*num_entries=*/4, /*offset=*/1024);
    Program program = MakeProgramFromSpec(fixture_device(), spec);
    ProgramRunArgs args;
    args.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(arena)}};
    args.dfb_run_overrides = {{.dfb = DFBSpecName{"fifo"}, .num_entries = 6}};
    SetProgramRunArgs(program, args);
    EXPECT_EQ(ArenaDFB(program).total_size(), 3072u);
    EXPECT_EQ(ArenaDFB(program).borrowed_addr_, ArenaBase(arena) + 1024);
}

TEST_F(SRAMArenaDFBTest, CPU_PartialResizeDoesNotRequireAllocationArgument) {
    // A partial resize can fill the remaining reservation without supplying the allocation again.
    SRAMArena arena = SRAMArena::allocate(fixture_device(), ArenaProperties(4096));
    ProgramSpec spec = ArenaDFBSpec(/*entry_size=*/512, /*num_entries=*/4, /*offset=*/1024);
    Program program = MakeProgramFromSpec(fixture_device(), spec);
    ProgramRunArgs args;
    args.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(arena)}};
    SetProgramRunArgs(program, args);
    ProgramRunArgs update;
    update.dfb_run_overrides = {{.dfb = DFBSpecName{"fifo"}, .num_entries = 6}};
    UpdateProgramRunArgs(program, update);
    EXPECT_EQ(ArenaDFB(program).total_size(), 3072u);
    EXPECT_EQ(ArenaDFB(program).borrowed_addr_, ArenaBase(arena) + 1024);
}

TEST_F(SRAMArenaDFBTest, CPU_FullResizeRejectsStorageBeyondOffsetLimit) {
    // A DFB that fits only when its offset is ignored must be rejected by a full argument set.
    SRAMArena arena = SRAMArena::allocate(fixture_device(), ArenaProperties(4096));
    ProgramSpec spec = ArenaDFBSpec(/*entry_size=*/512, /*num_entries=*/4, /*offset=*/1024);
    Program program = MakeProgramFromSpec(fixture_device(), spec);
    ProgramRunArgs args;
    args.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(arena)}};
    args.dfb_run_overrides = {{.dfb = DFBSpecName{"fifo"}, .num_entries = 8}};
    EXPECT_THROW(SetProgramRunArgs(program, args), std::runtime_error);
}

TEST_F(SRAMArenaDFBTest, CPU_PartialResizeRejectsStorageBeyondOffsetLimit) {
    // Omitting the allocation does not bypass the offset-inclusive resize bound.
    SRAMArena arena = SRAMArena::allocate(fixture_device(), ArenaProperties(4096));
    ProgramSpec spec = ArenaDFBSpec(/*entry_size=*/512, /*num_entries=*/4, /*offset=*/1024);
    Program program = MakeProgramFromSpec(fixture_device(), spec);
    ProgramRunArgs args;
    args.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(arena)}};
    SetProgramRunArgs(program, args);
    ProgramRunArgs update;
    update.dfb_run_overrides = {{.dfb = DFBSpecName{"fifo"}, .num_entries = 8}};
    EXPECT_THROW(UpdateProgramRunArgs(program, update), std::runtime_error);
}

TEST_F(SRAMArenaDFBTest, CPU_EntrySizeOverrideChecksReservationBound) {
    // An entry-size override must fit with the current entry count and allocation offset.
    SRAMArena arena = SRAMArena::allocate(fixture_device(), ArenaProperties(4096));
    ProgramSpec spec = ArenaDFBSpec(/*entry_size=*/512, /*num_entries=*/4, /*offset=*/1024);
    Program program = MakeProgramFromSpec(fixture_device(), spec);
    ProgramRunArgs args;
    args.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(arena)}};
    SetProgramRunArgs(program, args);
    ProgramRunArgs update;
    update.dfb_run_overrides = {{.dfb = DFBSpecName{"fifo"}, .entry_size = 1024}};
    EXPECT_THROW(UpdateProgramRunArgs(program, update), std::runtime_error);
}

TEST_F(SRAMArenaDFBTest, CPU_CombinedSizeOverridesUseResultingSize) {
    // Entry size and count overrides are checked together against the available storage.
    SRAMArena arena = SRAMArena::allocate(fixture_device(), ArenaProperties(4096));
    ProgramSpec spec = ArenaDFBSpec(/*entry_size=*/512, /*num_entries=*/4, /*offset=*/1024);
    Program program = MakeProgramFromSpec(fixture_device(), spec);
    ProgramRunArgs args;
    args.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(arena)}};
    args.dfb_run_overrides = {{.dfb = DFBSpecName{"fifo"}, .entry_size = 1536, .num_entries = 2}};
    SetProgramRunArgs(program, args);
    EXPECT_EQ(ArenaDFB(program).config.entry_size, 1536u);
    EXPECT_EQ(ArenaDFB(program).config.num_entries, 2u);
}

TEST_F(SRAMArenaDFBTest, CPU_LaterResizeUsesCurrentEntryCount) {
    // An omitted entry count retains its previous override during a later resize.
    SRAMArena arena = SRAMArena::allocate(fixture_device(), ArenaProperties(4096));
    ProgramSpec spec = ArenaDFBSpec(/*entry_size=*/512, /*num_entries=*/4, /*offset=*/1024);
    Program program = MakeProgramFromSpec(fixture_device(), spec);
    ProgramRunArgs args;
    args.advanced_options.sram_arena_args = {{SRAMArenaParamName{"arena"}, std::cref(arena)}};
    args.dfb_run_overrides = {{.dfb = DFBSpecName{"fifo"}, .num_entries = 2}};
    SetProgramRunArgs(program, args);
    ProgramRunArgs update;
    update.dfb_run_overrides = {{.dfb = DFBSpecName{"fifo"}, .entry_size = 1536}};
    UpdateProgramRunArgs(program, update);
    EXPECT_EQ(ArenaDFB(program).config.entry_size, 1536u);
    EXPECT_EQ(ArenaDFB(program).config.num_entries, 2u);
}

}  // namespace
}  // namespace tt::tt_metal::experimental
