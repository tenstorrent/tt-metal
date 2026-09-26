// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include <tt-logger/tt-logger.hpp>
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/operations/ccl/common/types/ccl_types.hpp"
#include "ttnn/operations/ccl/common/uops/ccl_command.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_common/reduce_scatter_program_utils.hpp"
#include <umd/device/types/xy_pair.hpp>

using ttnn::ccl::generate_slice_sequence_on_dim;
using ttnn::ccl::cmd::CclCommandArg;
using ttnn::ccl::cmd::CclCommandArgCode;
using ttnn::ccl::cmd::CclCommandCode;
using ttnn::ccl::cmd::CclCommandHeader;
using shape4d = ttnn::ccl::Shape4D<uint32_t>;

TEST(LineReduceScatter, EmitCclSendSliceSequenceCommands_8Slices_1x1x32x2048Tensor_Dim3_Slice0to7) {
    const std::size_t num_slices = 8;
    const std::int64_t start_slice_index = 0;
    const std::int64_t end_slice_index_exclusive = 8;
    const tt_xy_pair tensor_shape(64, 1);
    const tt_xy_pair worker_slice_shape(16, 1);
    const std::size_t scatter_dim = 3;
    const std::size_t worker_index = 0;
    auto const& slices = generate_slice_sequence_on_dim(
        tensor_shape,
        worker_slice_shape,
        scatter_dim,
        num_slices,
        start_slice_index,
        end_slice_index_exclusive,
        worker_index);

    std::vector<uint32_t> args;
    ASSERT_EQ(slices.size(), 8);
    ttnn::ccl::worker_detail::emit_ccl_send_slice_sequence_commands(slices, args);

    const std::size_t args_per_command_header = 1;
    const std::size_t args_per_command_arg_header = 1;

    const std::size_t args_per_full_tensor_field =
        CclCommandArg<CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES>::size_in_words();
    const std::size_t args_per_full_tensor_slice_command =
        args_per_command_header + args_per_command_arg_header + args_per_full_tensor_field;

    const std::size_t args_per_shape_field =
        CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES>::size_in_words();
    const std::size_t args_per_member_update =
        args_per_command_header + args_per_command_arg_header + args_per_shape_field;
    const std::size_t num_commands_with_single_field_update = num_slices - 1;

    ASSERT_EQ(
        args.size(),
        num_commands_with_single_field_update * args_per_member_update + args_per_full_tensor_slice_command);

    shape4d expected_tensor_slice_shape = shape4d(1, 1, 1, 8);

    log_info(tt::LogOp, "Commands");
    for (std::size_t i = 0; i < args.size(); i++) {
        log_info(tt::LogOp, "arg {}: {}", i, args[i]);
    }

    {  // Validate the first command
        std::size_t cmd_start_offset = 0;
        CclCommandHeader cmd_hdr = CclCommandHeader::from_uint32(args[cmd_start_offset]);
        CclCommandCode cmd_code = cmd_hdr.code;
        auto arg_count = cmd_hdr.arg_count;
        ASSERT_EQ(cmd_code, CclCommandCode::STREAM_TENSOR_TO_EDM);
        ASSERT_EQ(arg_count, 1);

        std::size_t arg_start_offset = cmd_start_offset + args_per_command_header;
        std::size_t fields_start = arg_start_offset + args_per_command_arg_header;
        std::size_t arg_offset = fields_start;
        ASSERT_EQ(args[arg_offset++], 1);
        ASSERT_EQ(args[arg_offset++], 1);
        ASSERT_EQ(args[arg_offset++], tensor_shape.y);
        ASSERT_EQ(args[arg_offset++], tensor_shape.x);

        ASSERT_EQ(args[arg_offset++], expected_tensor_slice_shape.w);
        ASSERT_EQ(args[arg_offset++], expected_tensor_slice_shape.z);
        ASSERT_EQ(args[arg_offset++], expected_tensor_slice_shape.y);
        ASSERT_EQ(args[arg_offset++], expected_tensor_slice_shape.x);
    }
}

// Worker-count heuristic input: the cores that stay on the grid after choose_worker_cores shifts every
// selected core by core_grid_offset. The regression behind #57507 / #57519: an 11x10 Blackhole grid
// with the reduce-scatter pushed below an 8-row matmul (offset (0, 8)) leaves 22 cores, not 110.
namespace {
tt::tt_metal::CoreRangeSet rectangular_grid(uint32_t num_cols, uint32_t num_rows) {
    return tt::tt_metal::CoreRangeSet(
        tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord(0, 0), tt::tt_metal::CoreCoord(num_cols - 1, num_rows - 1)));
}
}  // namespace

TEST(ReduceScatterDefaultWorkers, PlaceableCoreCountNoOffsetIsWholeGrid) {
    EXPECT_EQ(
        ttnn::experimental::ccl::count_worker_cores_placeable_after_offset(
            rectangular_grid(11, 10), tt::tt_metal::CoreCoord(0, 0)),
        110u);
}

TEST(ReduceScatterDefaultWorkers, PlaceableCoreCountRowOffsetKeepsBottomRows) {
    // Offset (0, 8): rows 0 and 1 shift onto rows 8 and 9, rows 2..9 shift off the grid.
    EXPECT_EQ(
        ttnn::experimental::ccl::count_worker_cores_placeable_after_offset(
            rectangular_grid(11, 10), tt::tt_metal::CoreCoord(0, 8)),
        22u);
    // Wormhole-shaped grid, same offset.
    EXPECT_EQ(
        ttnn::experimental::ccl::count_worker_cores_placeable_after_offset(
            rectangular_grid(8, 8), tt::tt_metal::CoreCoord(0, 6)),
        16u);
}

TEST(ReduceScatterDefaultWorkers, PlaceableCoreCountColumnAndDiagonalOffsets) {
    EXPECT_EQ(
        ttnn::experimental::ccl::count_worker_cores_placeable_after_offset(
            rectangular_grid(11, 10), tt::tt_metal::CoreCoord(1, 0)),
        100u);
    EXPECT_EQ(
        ttnn::experimental::ccl::count_worker_cores_placeable_after_offset(
            rectangular_grid(11, 10), tt::tt_metal::CoreCoord(1, 8)),
        20u);
}

TEST(ReduceScatterDefaultWorkers, PlaceableCoreCountOffsetPastTheGridIsZero) {
    EXPECT_EQ(
        ttnn::experimental::ccl::count_worker_cores_placeable_after_offset(
            rectangular_grid(11, 10), tt::tt_metal::CoreCoord(0, 10)),
        0u);
    EXPECT_EQ(
        ttnn::experimental::ccl::count_worker_cores_placeable_after_offset(
            rectangular_grid(11, 10), tt::tt_metal::CoreCoord(11, 0)),
        0u);
}

TEST(ReduceScatterDefaultWorkers, PlaceableCoreCountNonRectangularGrid) {
    // Two disjoint blocks: columns 0..7 over rows 0..9 and columns 8..10 over rows 0..5.
    tt::tt_metal::CoreRangeSet grid(std::vector<tt::tt_metal::CoreRange>{
        tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord(0, 0), tt::tt_metal::CoreCoord(7, 9)),
        tt::tt_metal::CoreRange(tt::tt_metal::CoreCoord(8, 0), tt::tt_metal::CoreCoord(10, 5))});
    EXPECT_EQ(ttnn::experimental::ccl::count_worker_cores_placeable_after_offset(grid, tt::tt_metal::CoreCoord(0, 0)), 98u);
    // Offset (0, 8): the tall block keeps its rows 0..1 (16 cores); the short block's rows shift to 8..13, all
    // outside it and outside the tall block's columns.
    EXPECT_EQ(ttnn::experimental::ccl::count_worker_cores_placeable_after_offset(grid, tt::tt_metal::CoreCoord(0, 8)), 16u);
    // Offset (0, 4): tall block rows 0..5 stay (48 cores); short block rows 0..1 land on its own rows 4..5 (6 cores).
    EXPECT_EQ(ttnn::experimental::ccl::count_worker_cores_placeable_after_offset(grid, tt::tt_metal::CoreCoord(0, 4)), 54u);
}
