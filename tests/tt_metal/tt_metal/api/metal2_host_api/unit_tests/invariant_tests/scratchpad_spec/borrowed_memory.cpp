// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <stdexcept>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <tt_stl/fmt.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalValidProgramSpec;
using test_helpers::ProgramSpecTestQuasar;

TEST_F(ProgramSpecTestQuasar, RawAllocationScratchpadExactFitAtZeroOffset) {
    // A scratchpad may occupy the allocation’s entire per-node extent.
    const uint32_t alignment = mesh_device_->allocator()->get_alignment(BufferType::L1);
    auto spec = MakeMinimalValidProgramSpec();
    spec.advanced_options.raw_allocation_parameters = {RawAllocationParameter{
        .unique_id = RawAllocationParamName{"arena"},
        .properties =
            RangeLockstepAllocationProperties{
                .banks_to_be_allocated_on = NodeRangeSet{NodeRange{NodeCoord{0, 0}, NodeCoord{0, 0}}},
                .per_bank_size = 2 * alignment,
            },
    }};
    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch"}, .size_per_node = 2 * alignment}};
    spec.scratchpads[0].advanced_options.borrowed_from =
        ScratchpadAdvancedOptions::RawAllocationBorrowing{.allocation = RawAllocationParamName{"arena"}, .offset = 0};
    spec.kernels[0].scratchpad_bindings = {KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"scratch"}, .accessor_name = "scratch"}};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, RawAllocationScratchpadExactFitAtNonzeroOffset) {
    // An aligned offset is allowed, and the scratchpad may end exactly at the allocation boundary.
    const uint32_t alignment = mesh_device_->allocator()->get_alignment(BufferType::L1);
    auto spec = MakeMinimalValidProgramSpec();
    spec.advanced_options.raw_allocation_parameters = {RawAllocationParameter{
        .unique_id = RawAllocationParamName{"arena"},
        .properties =
            RangeLockstepAllocationProperties{
                .banks_to_be_allocated_on = NodeRangeSet{NodeRange{NodeCoord{0, 0}, NodeCoord{0, 0}}},
                .per_bank_size = 3 * alignment,
            },
    }};
    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch"}, .size_per_node = 2 * alignment}};
    spec.scratchpads[0].advanced_options.borrowed_from = ScratchpadAdvancedOptions::RawAllocationBorrowing{
        .allocation = RawAllocationParamName{"arena"}, .offset = alignment};
    spec.kernels[0].scratchpad_bindings = {KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"scratch"}, .accessor_name = "scratch"}};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, RawAllocationScratchpadExtraAllocationNodes) {
    // The allocation may cover more nodes than the scratchpad occupies.
    const uint32_t alignment = mesh_device_->allocator()->get_alignment(BufferType::L1);
    auto spec = MakeMinimalValidProgramSpec();
    spec.advanced_options.raw_allocation_parameters = {RawAllocationParameter{
        .unique_id = RawAllocationParamName{"arena"},
        .properties =
            RangeLockstepAllocationProperties{
                .banks_to_be_allocated_on = NodeRangeSet{NodeRange{NodeCoord{0, 0}, NodeCoord{1, 0}}},
                .per_bank_size = 2 * alignment,
            },
    }};
    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch"}, .size_per_node = 2 * alignment}};
    spec.scratchpads[0].advanced_options.borrowed_from =
        ScratchpadAdvancedOptions::RawAllocationBorrowing{.allocation = RawAllocationParamName{"arena"}, .offset = 0};
    spec.kernels[0].scratchpad_bindings = {KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"scratch"}, .accessor_name = "scratch"}};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, RawAllocationScratchpadMissingOccupiedNode) {
    // The allocation must cover every node occupied by the scratchpad.
    const uint32_t alignment = mesh_device_->allocator()->get_alignment(BufferType::L1);
    auto spec = MakeMinimalValidProgramSpec();
    spec.advanced_options.raw_allocation_parameters = {RawAllocationParameter{
        .unique_id = RawAllocationParamName{"arena"},
        .properties =
            RangeLockstepAllocationProperties{
                .banks_to_be_allocated_on = NodeRangeSet{NodeRange{NodeCoord{0, 0}, NodeCoord{0, 0}}},
                .per_bank_size = 2 * alignment,
            },
    }};
    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch"}, .size_per_node = 2 * alignment}};
    spec.scratchpads[0].advanced_options.borrowed_from =
        ScratchpadAdvancedOptions::RawAllocationBorrowing{.allocation = RawAllocationParamName{"arena"}, .offset = 0};
    spec.kernels[0].scratchpad_bindings = {KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"scratch"}, .accessor_name = "scratch"}};
    spec.work_units[0].target_nodes = NodeRange{NodeCoord{0, 0}, NodeCoord{1, 0}};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("ScratchpadSpec 'scratch' borrows from RawAllocationParameter 'arena', which does not "
                                 "cover every occupied node.")));
}

TEST_F(ProgramSpecTestQuasar, RawAllocationScratchpadExceedsCapacity) {
    // The scratchpad’s per-node size cannot exceed the allocation’s per-node capacity.
    const uint32_t alignment = mesh_device_->allocator()->get_alignment(BufferType::L1);
    auto spec = MakeMinimalValidProgramSpec();
    spec.advanced_options.raw_allocation_parameters = {RawAllocationParameter{
        .unique_id = RawAllocationParamName{"arena"},
        .properties =
            RangeLockstepAllocationProperties{
                .banks_to_be_allocated_on = NodeRangeSet{NodeRange{NodeCoord{0, 0}, NodeCoord{0, 0}}},
                .per_bank_size = 2 * alignment - 1,
            },
    }};
    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch"}, .size_per_node = 2 * alignment}};
    spec.scratchpads[0].advanced_options.borrowed_from =
        ScratchpadAdvancedOptions::RawAllocationBorrowing{.allocation = RawAllocationParamName{"arena"}, .offset = 0};
    spec.kernels[0].scratchpad_bindings = {KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"scratch"}, .accessor_name = "scratch"}};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(fmt::format(
            "ScratchpadSpec 'scratch' borrowing offset {} + size_per_node {} exceeds RawAllocationParameter 'arena' "
            "per_bank_size {}.",
            0,
            2 * alignment,
            2 * alignment - 1))));
}

TEST_F(ProgramSpecTestQuasar, RawAllocationScratchpadOffsetCausesOverrun) {
    // The capacity check includes the offset, even when the scratchpad’s size alone fits.
    const uint32_t alignment = mesh_device_->allocator()->get_alignment(BufferType::L1);
    auto spec = MakeMinimalValidProgramSpec();
    spec.advanced_options.raw_allocation_parameters = {RawAllocationParameter{
        .unique_id = RawAllocationParamName{"arena"},
        .properties =
            RangeLockstepAllocationProperties{
                .banks_to_be_allocated_on = NodeRangeSet{NodeRange{NodeCoord{0, 0}, NodeCoord{0, 0}}},
                .per_bank_size = 3 * alignment - 1,
            },
    }};
    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch"}, .size_per_node = 2 * alignment}};
    spec.scratchpads[0].advanced_options.borrowed_from = ScratchpadAdvancedOptions::RawAllocationBorrowing{
        .allocation = RawAllocationParamName{"arena"}, .offset = alignment};
    spec.kernels[0].scratchpad_bindings = {KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"scratch"}, .accessor_name = "scratch"}};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(fmt::format(
            "ScratchpadSpec 'scratch' borrowing offset {} + size_per_node {} exceeds RawAllocationParameter 'arena' "
            "per_bank_size {}.",
            alignment,
            2 * alignment,
            3 * alignment - 1))));
}

TEST_F(ProgramSpecTestQuasar, RawAllocationScratchpadMisalignedOffset) {
    // The borrowing offset must satisfy L1 alignment, even when capacity is sufficient.
    const uint32_t alignment = mesh_device_->allocator()->get_alignment(BufferType::L1);
    ASSERT_GT(alignment, 1u);
    auto spec = MakeMinimalValidProgramSpec();
    spec.advanced_options.raw_allocation_parameters = {RawAllocationParameter{
        .unique_id = RawAllocationParamName{"arena"},
        .properties =
            RangeLockstepAllocationProperties{
                .banks_to_be_allocated_on = NodeRangeSet{NodeRange{NodeCoord{0, 0}, NodeCoord{0, 0}}},
                .per_bank_size = 4 * alignment,
            },
    }};
    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch"}, .size_per_node = 2 * alignment}};
    spec.scratchpads[0].advanced_options.borrowed_from =
        ScratchpadAdvancedOptions::RawAllocationBorrowing{.allocation = RawAllocationParamName{"arena"}, .offset = 1};
    spec.kernels[0].scratchpad_bindings = {KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"scratch"}, .accessor_name = "scratch"}};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(fmt::format(
            "ScratchpadSpec 'scratch' borrowing offset 1 must be a multiple of L1 alignment {}.", alignment))));
}

TEST_F(ProgramSpecTestQuasar, RawAllocationScratchpadAllBankLockstep) {
    // Scratchpad borrowing accepts the all-bank property alternative as well as range-lockstep.
    const uint32_t alignment = mesh_device_->allocator()->get_alignment(BufferType::L1);
    auto spec = MakeMinimalValidProgramSpec();
    spec.advanced_options.raw_allocation_parameters = {RawAllocationParameter{
        .unique_id = RawAllocationParamName{"arena"},
        .properties = LockstepAllocationProperties{.per_bank_size = 2 * alignment},
    }};
    spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch"}, .size_per_node = 2 * alignment}};
    spec.scratchpads[0].advanced_options.borrowed_from =
        ScratchpadAdvancedOptions::RawAllocationBorrowing{.allocation = RawAllocationParamName{"arena"}, .offset = 0};
    spec.kernels[0].scratchpad_bindings = {KernelSpec::ScratchpadBinding{
        .scratchpad_spec_name = ScratchpadSpecName{"scratch"}, .accessor_name = "scratch"}};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
