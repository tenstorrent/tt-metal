// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "ttnn/cpp/ttnn/operations/ccl/reduce_scatter/host/reduce_scatter_worker_builder.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_command.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/types/ccl_types.hpp"

#include <vector>
#include <cstdint>

using ttnn::ccl::cmd::CclCommandArg;
using ttnn::ccl::cmd::CclCommandArgCode;
using ttnn::ccl::cmd::CclCommandHeader;
using ttnn::ccl::cmd::CclCommandCode;
using ttnn::ccl::generate_slice_sequence_on_dim;
using shape4d = ttnn::ccl::Shape4D<uint32_t>;
TEST(LineReduceScatter, EmitCclSendSliceSequenceCommands_8Slices_1x1x32x2048Tensor_Dim3_Slice0to7)
{
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
        worker_index
    );

    std::vector<uint32_t> args;
    ASSERT_EQ(slices.size(), 8);
    ttnn::ccl::reduce_scatter_detail::emit_ccl_send_slice_sequence_commands(slices, args);

    const std::size_t args_per_command_header = 1;
    const std::size_t args_per_command_arg_header = 1;

    const std::size_t args_per_full_tensor_field = CclCommandArg<CclCommandArgCode::SET_FULL_TENSOR_SLICE_SPEC_IN_PAGES>::size_in_words();
    const std::size_t args_per_full_tensor_slice_command = args_per_command_header + args_per_command_arg_header + args_per_full_tensor_field;

    const std::size_t args_per_shape_field = CclCommandArg<CclCommandArgCode::SET_TENSOR_SLICE_OFFSET_IN_PAGES>::size_in_words();
    const std::size_t args_per_member_update = args_per_command_header + args_per_command_arg_header + args_per_shape_field;
    const std::size_t num_commands_with_single_field_update = num_slices - 1;

    ASSERT_EQ(args.size(), num_commands_with_single_field_update * args_per_member_update + args_per_full_tensor_slice_command);

    shape4d expected_tensor_slice_shape = shape4d(1, 1, 1, 8);

    log_info(tt::LogOp, "Commands");
    for (std::size_t i = 0; i < args.size(); i++) {
        log_info(tt::LogOp, "arg {}: {}", i, args[i]);
    }


    { // Validate the first command
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
