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

using test_helpers::MakeBorrowedDFBProgramSpec;
using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::ProgramSpecTestQuasar;

TEST_F(ProgramSpecTestQuasar, CPU_SRAMArenaDFBUndeclaredName) {
    // A DFB’s SRAM arena name must resolve to a declared `SRAMArenaParameter`.
    const uint32_t alignment = mesh_device_->allocator()->get_alignment(BufferType::L1);
    auto spec = MakeBorrowedDFBProgramSpec("tensor", BufferType::L1, alignment, 2, false);
    spec.tensor_parameters.clear();
    spec.advanced_options.sram_arena_parameters = {SRAMArenaParameter{
        .unique_id = SRAMArenaParamName{"arena"},
        .properties =
            RangeLockstepSRAMArenaProperties{
                .banks_to_be_allocated_on = NodeRangeSet{NodeRange{NodeCoord{0, 0}, NodeCoord{0, 0}}},
                .per_bank_size = 2 * alignment,
            },
    }};
    spec.dataflow_buffers[0].borrowed_from.reset();
    spec.dataflow_buffers[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"missing"}};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("DFB 'dfb' allocates from unknown SRAMArenaParameter 'missing'.")));
}

TEST_F(ProgramSpecTestQuasar, CPU_SRAMArenaScratchpadUndeclaredName) {
    // A scratchpad’s SRAM arena name must resolve to a declared `SRAMArenaParameter`.
    const uint32_t alignment = mesh_device_->allocator()->get_alignment(BufferType::L1);
    auto spec = MakeMinimalValidProgramSpec();
    spec.advanced_options.sram_arena_parameters = {SRAMArenaParameter{
        .unique_id = SRAMArenaParamName{"arena"},
        .properties =
            RangeLockstepSRAMArenaProperties{
                .banks_to_be_allocated_on = NodeRangeSet{NodeRange{NodeCoord{0, 0}, NodeCoord{0, 0}}},
                .per_bank_size = 2 * alignment,
            },
    }};
    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch"}, .size_per_node = 2 * alignment}};
    spec.scratchpads[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"missing"}, .offset = 0};
    spec.kernels[0].scratchpad_bindings = {KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"scratch"}, .accessor_name = "scratch"}};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("ScratchpadSpec 'scratch' allocates from unknown SRAMArenaParameter 'missing'.")));
}

TEST_F(ProgramSpecTestQuasar, CPU_SRAMArenaUnusedLockstepParameterFails) {
    // Program construction rejects an unused lockstep arena parameter.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = MakeMinimalValidProgramSpec();
    spec.advanced_options.sram_arena_parameters = {
        {.unique_id = SRAMArenaParamName{"unused"}, .properties = LockstepSRAMArenaProperties{.per_bank_size = 4096}}};
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(fixture_device, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("SRAMArenaParameter 'unused' is not used by any DFB or scratchpad")));
}

TEST_F(ProgramSpecTestQuasar, CPU_SRAMArenaUnusedRangeLockstepParameterFails) {
    // Program construction rejects an unused range-lockstep arena parameter.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = MakeMinimalValidProgramSpec();
    const RangeLockstepSRAMArenaProperties properties{
        .banks_to_be_allocated_on = NodeRangeSet{NodeRange({0, 0}, {0, 0})}, .per_bank_size = 4096};
    spec.advanced_options.sram_arena_parameters = {
        {.unique_id = SRAMArenaParamName{"unused"}, .properties = properties}};
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(fixture_device, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("SRAMArenaParameter 'unused' is not used by any DFB or scratchpad")));
}

TEST_F(ProgramSpecTestQuasar, CPU_SRAMArenaUsedOnlyByDFBSucceeds) {
    // A DFB allocation is sufficient to use an arena parameter.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = MakeMinimalValidProgramSpec();
    spec.advanced_options.sram_arena_parameters = {
        {.unique_id = SRAMArenaParamName{"arena"}, .properties = LockstepSRAMArenaProperties{.per_bank_size = 4096}}};
    spec.dataflow_buffers[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"arena"}};
    EXPECT_NO_THROW(MakeProgramFromSpec(fixture_device, spec));
}

TEST_F(ProgramSpecTestQuasar, CPU_SRAMArenaUnusedParameterBesideUsedParameterFails) {
    // Using one arena does not permit another arena parameter to remain unused.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = MakeMinimalValidProgramSpec();
    spec.advanced_options.sram_arena_parameters = {
        {.unique_id = SRAMArenaParamName{"used"}, .properties = LockstepSRAMArenaProperties{.per_bank_size = 4096}},
        {.unique_id = SRAMArenaParamName{"unused"}, .properties = LockstepSRAMArenaProperties{.per_bank_size = 4096}}};
    spec.dataflow_buffers[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"used"}};
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(fixture_device, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("SRAMArenaParameter 'unused' is not used by any DFB or scratchpad")));
}

TEST_F(ProgramSpecTestQuasar, CPU_SRAMArenaUnboundScratchpadFails) {
    // Program construction rejects an arena-backed scratchpad that no kernel binds.
    distributed::MeshDevice& fixture_device = *mesh_device_;
    auto spec = MakeMinimalValidProgramSpec();
    spec.advanced_options.sram_arena_parameters = {
        {.unique_id = SRAMArenaParamName{"arena"}, .properties = LockstepSRAMArenaProperties{.per_bank_size = 4096}}};
    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"orphan_scratch"}, .size_per_node = 1024}};
    spec.scratchpads[0].advanced_options.allocation_override =
        SRAMArenaAllocation{.arena = SRAMArenaParamName{"arena"}};
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(fixture_device, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("ScratchpadSpec 'orphan_scratch' is declared but not bound by any kernel.")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
