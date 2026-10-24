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

TEST_F(ProgramSpecTestQuasar, RawAllocationParameterValidLockstep) {
    // A positive per-bank size is accepted.
    const uint32_t alignment = mesh_device_->allocator()->get_alignment(BufferType::L1);
    auto spec = MakeMinimalValidProgramSpec();
    spec.advanced_options.raw_allocation_parameters = {RawAllocationParameter{
        .unique_id = RawAllocationParamName{"arena"},
        .properties = LockstepAllocationProperties{.per_bank_size = alignment},
    }};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, RawAllocationParameterValidRangeLockstep) {
    // A positive per-bank size and nonempty node set are accepted.
    const uint32_t alignment = mesh_device_->allocator()->get_alignment(BufferType::L1);
    auto spec = MakeMinimalValidProgramSpec();
    spec.advanced_options.raw_allocation_parameters = {RawAllocationParameter{
        .unique_id = RawAllocationParamName{"arena"},
        .properties =
            RangeLockstepAllocationProperties{
                .banks_to_be_allocated_on = NodeRangeSet{NodeRange{NodeCoord{0, 0}, NodeCoord{0, 0}}},
                .per_bank_size = alignment,
            },
    }};

    EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, spec));
}

TEST_F(ProgramSpecTestQuasar, RawAllocationParameterZeroLockstepSize) {
    // A lockstep allocation parameter cannot declare zero bytes per bank.
    auto spec = MakeMinimalValidProgramSpec();
    spec.advanced_options.raw_allocation_parameters = {RawAllocationParameter{
        .unique_id = RawAllocationParamName{"arena"},
        .properties = LockstepAllocationProperties{.per_bank_size = 0},
    }};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
            "RawAllocationParameter 'arena' has per_bank_size = 0; per_bank_size must be greater than zero.")));
}

TEST_F(ProgramSpecTestQuasar, RawAllocationParameterZeroRangeLockstepSize) {
    // A range-lockstep allocation parameter cannot declare zero bytes per bank.
    auto spec = MakeMinimalValidProgramSpec();
    spec.advanced_options.raw_allocation_parameters = {RawAllocationParameter{
        .unique_id = RawAllocationParamName{"arena"},
        .properties =
            RangeLockstepAllocationProperties{
                .banks_to_be_allocated_on = NodeRangeSet{NodeRange{NodeCoord{0, 0}, NodeCoord{0, 0}}},
                .per_bank_size = 0,
            },
    }};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
            "RawAllocationParameter 'arena' has per_bank_size = 0; per_bank_size must be greater than zero.")));
}

TEST_F(ProgramSpecTestQuasar, RawAllocationParameterEmptyRangeLockstepNodes) {
    // A range-lockstep allocation parameter must cover at least one node.
    const uint32_t alignment = mesh_device_->allocator()->get_alignment(BufferType::L1);
    auto spec = MakeMinimalValidProgramSpec();
    spec.advanced_options.raw_allocation_parameters = {RawAllocationParameter{
        .unique_id = RawAllocationParamName{"arena"},
        .properties =
            RangeLockstepAllocationProperties{
                .banks_to_be_allocated_on = NodeRangeSet{},
                .per_bank_size = alignment,
            },
    }};

    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
            "RawAllocationParameter 'arena' has empty banks_to_be_allocated_on; at least one node is required.")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
