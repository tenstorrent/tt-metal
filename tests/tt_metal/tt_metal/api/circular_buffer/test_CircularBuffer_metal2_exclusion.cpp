// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/host_api.hpp>

#include "distributed/mesh_device_impl.hpp"
#include "impl/metal2_host_api/program_spec/construction/construct_program.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::ProgramSpecTestBlackhole;

TEST_F(ProgramSpecTestBlackhole, CPU_RejectsMixedCircularBufferAndDFBAllocations) {
    const CoreRangeSet left(CoreRange({0, 0}));
    const CoreRangeSet right(CoreRange({1, 0}));
    auto cb_config = CircularBufferConfig(1024, {{0, DataFormat::Float16_b}}).set_page_size(0, 1024);
    const dfb::DataflowBufferConfig dfb_config{.entry_size = 1024, .num_entries = 4};
    auto cb_first = CreateProgram(*mesh_device_);
    ASSERT_NO_THROW(cb_first.impl().add_circular_buffer(left, cb_config));
    EXPECT_THAT(
        [&] { cb_first.impl().add_dataflow_buffer(right, dfb_config); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Cannot add dataflow buffer to a program with circular buffers")));
    auto dfb_first = CreateProgram(*mesh_device_);
    ASSERT_NO_THROW(dfb_first.impl().add_dataflow_buffer(right, dfb_config));
    EXPECT_THAT(
        [&] { dfb_first.impl().add_circular_buffer(left, cb_config); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Cannot add circular buffer to a program that already has dataflow buffers")));
}

TEST_F(ProgramSpecTestBlackhole, CPU_RejectsCircularBufferInScratchpadOnlySpecProgram) {
    ProgramSpec spec;
    spec.kernels = {test_helpers::MakeMinimalGen1DMKernel("dm")};
    spec.scratchpads = {{.unique_id = ScratchpadSpecName{"scratch"}, .size_per_node = 256}};
    spec.kernels[0].scratchpad_bindings = {
        {.scratchpad_spec_name = ScratchpadSpecName{"scratch"}, .accessor_name = "s"}};
    spec.work_units = {test_helpers::MakeMinimalWorkUnit("work", NodeCoord{0, 0}, {"dm"})};
    // Keep the program uncompiled to exercise the spec/legacy-CB exclusion directly.
    auto program = BuildProgram(*mesh_device_, spec, CollectSpecData(spec), mesh_device_->impl().metal_context());
    ASSERT_FALSE(program.impl().is_compiled());
    ASSERT_TRUE(program.impl().created_from_spec());
    ASSERT_TRUE(program.impl().dataflow_buffers().empty());
    auto config = CircularBufferConfig(1024, {{0, DataFormat::Float16_b}}).set_page_size(0, 1024);
    EXPECT_THAT(
        [&] { CreateCircularBuffer(program, CoreRangeSet(CoreRange({0, 0})), config); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Cannot add a legacy circular buffer to a Metal 2.0 Program")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
