// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <stdexcept>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::ProgramSpecTestQuasar;

ProgramSpec MakeProgramSpecWithArenaScratchpadForTest() {
    auto spec = MakeMinimalValidProgramSpec();
    spec.scratchpads = {{.unique_id = ScratchpadSpecName{"scratch"}, .size_per_node = 16}};
    spec.scratchpads[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"arena"}};
    spec.kernels[0].scratchpad_bindings = {
        {.scratchpad_spec_name = ScratchpadSpecName{"scratch"}, .accessor_name = "scratch"}};
    return spec;
}

TEST_F(ProgramSpecTestQuasar, CPU_SRAMArenaParameterValidLockstep) {
    // A scratchpad can use a lockstep arena parameter with a positive per-bank size.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    const uint32_t alignment = fixture_device.allocator()->get_alignment(BufferType::L1);
    auto spec = MakeProgramSpecWithArenaScratchpadForTest();
    spec.advanced_options.sram_arena_parameters = {
        {.unique_id = SRAMArenaParamName{"arena"},
         .properties = LockstepSRAMArenaProperties{.per_bank_size = alignment}}};
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_SRAMArenaParameterValidRangeLockstep) {
    // A scratchpad can use a range-lockstep arena parameter covering its node.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    const uint32_t alignment = fixture_device.allocator()->get_alignment(BufferType::L1);
    auto spec = MakeProgramSpecWithArenaScratchpadForTest();
    const RangeLockstepSRAMArenaProperties properties{
        .banks_to_be_allocated_on = NodeRangeSet{NodeRange({0, 0}, {0, 0})}, .per_bank_size = alignment};
    spec.advanced_options.sram_arena_parameters = {
        {.unique_id = SRAMArenaParamName{"arena"}, .properties = properties}};
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_SRAMArenaParameterZeroLockstepSize) {
    // A lockstep allocation parameter cannot declare zero bytes per bank.
    auto spec = MakeMinimalValidProgramSpec();
    spec.advanced_options.sram_arena_parameters = {SRAMArenaParameter{
        .unique_id = SRAMArenaParamName{"arena"},
        .properties = LockstepSRAMArenaProperties{.per_bank_size = 0},
    }};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
            "SRAMArenaParameter 'arena' has per_bank_size = 0; per_bank_size must be greater than zero.")));
}

TEST_F(ProgramSpecTestQuasar, CPU_SRAMArenaParameterZeroRangeLockstepSize) {
    // A range-lockstep allocation parameter cannot declare zero bytes per bank.
    auto spec = MakeMinimalValidProgramSpec();
    spec.advanced_options.sram_arena_parameters = {SRAMArenaParameter{
        .unique_id = SRAMArenaParamName{"arena"},
        .properties =
            RangeLockstepSRAMArenaProperties{
                .banks_to_be_allocated_on = NodeRangeSet{NodeRange{NodeCoord{0, 0}, NodeCoord{0, 0}}},
                .per_bank_size = 0,
            },
    }};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
            "SRAMArenaParameter 'arena' has per_bank_size = 0; per_bank_size must be greater than zero.")));
}

TEST_F(ProgramSpecTestQuasar, CPU_SRAMArenaParameterEmptyRangeLockstepNodes) {
    // A range-lockstep allocation parameter must cover at least one node.
    const uint32_t alignment = mesh_device_->allocator()->get_alignment(BufferType::L1);
    auto spec = MakeMinimalValidProgramSpec();
    spec.advanced_options.sram_arena_parameters = {SRAMArenaParameter{
        .unique_id = SRAMArenaParamName{"arena"},
        .properties =
            RangeLockstepSRAMArenaProperties{
                .banks_to_be_allocated_on = NodeRangeSet{},
                .per_bank_size = alignment,
            },
    }};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
            "SRAMArenaParameter 'arena' has empty banks_to_be_allocated_on; at least one node is required.")));
}

TEST_F(ProgramSpecTestQuasar, CPU_SRAMArenaParameterZeroSizeUsedByScratchpad) {
    // A zero-size arena that a scratchpad allocates from is reported as a zero-size arena.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = MakeProgramSpecWithArenaScratchpadForTest();
    spec.advanced_options.sram_arena_parameters = {
        {.unique_id = SRAMArenaParamName{"arena"}, .properties = LockstepSRAMArenaProperties{.per_bank_size = 0}}};
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(fixture_device, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
            "SRAMArenaParameter 'arena' has per_bank_size = 0; per_bank_size must be greater than zero.")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
