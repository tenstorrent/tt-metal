// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_command.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/types/ccl_types.hpp"

#include "gtest/gtest.h"

#include <limits>
#include <numeric>
#include <ranges>

using ttnn::ccl::Shape4D;
using ttnn::ccl::cmd::tensor_shape_command_arg_t;
using ttnn::ccl::cmd::tensor_slice_shape_command_arg_t;
using ttnn::ccl::cmd::tensor_slice_offset_command_arg_t;
using ttnn::ccl::cmd::worker_start_offset_command_arg_t;
using ttnn::ccl::cmd::worker_pages_command_arg_t;
using ttnn::ccl::cmd::full_tensor_command_arg_t;
using ttnn::ccl::cmd::CclCommandTensor;

const Shape4D<uint32_t> uninitialized_test_shape = {
    std::numeric_limits<uint32_t>::max(),
    std::numeric_limits<uint32_t>::max(),
    std::numeric_limits<uint32_t>::max(),
    std::numeric_limits<uint32_t>::max()};

// tensor shape
TEST(CclCommandArgGenerator, PackTensorShapeArg) {
    constexpr std::size_t size_in_words = tensor_shape_command_arg_t::size_in_words();
    ASSERT_EQ(size_in_words, 4);
    std::array<uint32_t, size_in_words> args;
    std::ranges::fill(args, std::numeric_limits<uint32_t>::max());
    Shape4D<uint32_t> test_shape = {1,2,3,4};
    tensor_shape_command_arg_t::pack_to(args.data(), test_shape);
    ASSERT_EQ(args[0], 1);
    ASSERT_EQ(args[1], 2);
    ASSERT_EQ(args[2], 3);
    ASSERT_EQ(args[3], 4);
}

TEST(CclCommandArgGenerator, UnpackTensorShapeArg) {
    constexpr std::size_t size_in_words = tensor_shape_command_arg_t::size_in_words();
    ASSERT_EQ(size_in_words, 4);
    std::array<uint32_t, tensor_shape_command_arg_t::size_in_words()> args = {1,2,3,4};
    Shape4D<uint32_t> test_shape = uninitialized_test_shape;
    tensor_shape_command_arg_t::unpack(args.data(), test_shape);

    ASSERT_EQ(test_shape.w, 1);
    ASSERT_EQ(test_shape.z, 2);
    ASSERT_EQ(test_shape.y, 3);
    ASSERT_EQ(test_shape.x, 4);
}

// tensor slice
TEST(CclCommandArgGenerator, PackTensorSliceShapeArg) {
    std::array<uint32_t, tensor_slice_shape_command_arg_t::size_in_words()> args;
    std::ranges::fill(args, std::numeric_limits<uint32_t>::max());
    constexpr std::size_t size_in_words = tensor_slice_shape_command_arg_t::size_in_words();
    ASSERT_EQ(size_in_words, 4);
    Shape4D<uint32_t> test_shape = {1,2,3,4};
    tensor_slice_shape_command_arg_t::pack_to(args.data(), test_shape);
    ASSERT_EQ(args[0], 1);
    ASSERT_EQ(args[1], 2);
    ASSERT_EQ(args[2], 3);
    ASSERT_EQ(args[3], 4);
}

TEST(CclCommandArgGenerator, UnpackTensorSliceShapeArg) {
    std::array<uint32_t, tensor_slice_shape_command_arg_t::size_in_words()> args = {1,2,3,4};
    constexpr std::size_t size_in_words = tensor_slice_shape_command_arg_t::size_in_words();
    ASSERT_EQ(size_in_words, 4);
    Shape4D<uint32_t> test_shape = uninitialized_test_shape;
    tensor_slice_shape_command_arg_t::unpack(args.data(), test_shape);
    ASSERT_EQ(test_shape.w, 1);
    ASSERT_EQ(test_shape.z, 2);
    ASSERT_EQ(test_shape.y, 3);
    ASSERT_EQ(test_shape.x, 4);
}

// tensor slice offset
TEST(CclCommandArgGenerator, PackTensorSliceOffsetArg) {
    std::array<uint32_t, tensor_slice_offset_command_arg_t::size_in_words()> args;
    std::ranges::fill(args, std::numeric_limits<uint32_t>::max());
    constexpr std::size_t size_in_words = tensor_slice_offset_command_arg_t::size_in_words();
    ASSERT_EQ(size_in_words, 4);
    Shape4D<uint32_t> test_shape = {1,2,3,4};
    tensor_slice_offset_command_arg_t::pack_to(args.data(), test_shape);
    ASSERT_EQ(args[0], 1);
    ASSERT_EQ(args[1], 2);
    ASSERT_EQ(args[2], 3);
    ASSERT_EQ(args[3], 4);
}

TEST(CclCommandArgGenerator, UnpackTensorSliceOffsetArg) {
    std::array<uint32_t, tensor_slice_offset_command_arg_t::size_in_words()> args = {1,2,3,4};
    constexpr std::size_t size_in_words = tensor_slice_offset_command_arg_t::size_in_words();
    ASSERT_EQ(size_in_words, 4);
    Shape4D<uint32_t> test_shape = uninitialized_test_shape;
    tensor_slice_offset_command_arg_t::unpack(args.data(), test_shape);
    ASSERT_EQ(test_shape.w, 1);
    ASSERT_EQ(test_shape.z, 2);
    ASSERT_EQ(test_shape.y, 3);
    ASSERT_EQ(test_shape.x, 4);
}

// worker start offset in slice
TEST(CclCommandArgGenerator, PackWorkerStartOffsetInSliceArg) {
    std::array<uint32_t, worker_start_offset_command_arg_t::size_in_words()> args;
    std::ranges::fill(args, std::numeric_limits<uint32_t>::max());
    constexpr std::size_t size_in_words = worker_start_offset_command_arg_t::size_in_words();
    ASSERT_EQ(size_in_words, 4);
    Shape4D<uint32_t> test_shape = {1,2,3,4};
    worker_start_offset_command_arg_t::pack_to(args.data(), test_shape);
    ASSERT_EQ(args[0], 1);
    ASSERT_EQ(args[1], 2);
    ASSERT_EQ(args[2], 3);
    ASSERT_EQ(args[3], 4);
}

TEST(CclCommandArgGenerator, UnpackWorkerStartOffsetInSliceArg) {
    std::array<uint32_t, worker_start_offset_command_arg_t::size_in_words()> args = {1,2,3,4};
    constexpr std::size_t size_in_words = worker_start_offset_command_arg_t::size_in_words();
    ASSERT_EQ(size_in_words, 4);
    Shape4D<uint32_t> test_shape = uninitialized_test_shape;
    worker_start_offset_command_arg_t::unpack(args.data(), test_shape);
    ASSERT_EQ(test_shape.w, 1);
    ASSERT_EQ(test_shape.z, 2);
    ASSERT_EQ(test_shape.y, 3);
    ASSERT_EQ(test_shape.x, 4);
}

// worker pages per slice
TEST(CclCommandArgGenerator, PackWorkerPagesPerSliceArg) {
    std::array<uint32_t, worker_pages_command_arg_t::size_in_words()> args;
    std::ranges::fill(args, std::numeric_limits<uint32_t>::max());
    constexpr std::size_t size_in_words = worker_pages_command_arg_t::size_in_words();
    ASSERT_EQ(size_in_words, 1);
    uint32_t test_value = 1;
    worker_pages_command_arg_t::pack_to(args.data(), test_value);
    ASSERT_EQ(args[0], 1);
}

TEST(CclCommandArgGenerator, UnpackWorkerPagesPerSliceArg) {
    std::array<uint32_t, worker_pages_command_arg_t::size_in_words()> args = {1};
    constexpr std::size_t size_in_words = worker_pages_command_arg_t::size_in_words();
    ASSERT_EQ(size_in_words, 1);
    uint32_t test_value = 0;
    worker_pages_command_arg_t::unpack(args.data(), test_value);
    ASSERT_EQ(test_value, 1);
}

// full tensor
TEST(CclCommandArgGenerator, PackFullTensorArg) {
    constexpr std::size_t size_in_words = full_tensor_command_arg_t::size_in_words();
    ASSERT_EQ(size_in_words, 17);
    std::array<uint32_t, full_tensor_command_arg_t::size_in_words()> args;
    std::ranges::fill(args, std::numeric_limits<uint32_t>::max());

    CclCommandTensor test_tensor = {
        {0,1,2,3},
        {4,5,6,7},
        {8,9,10,11},
        {12,13,14,15},
        16
    };
    full_tensor_command_arg_t::pack_to(args.data(), test_tensor);
    for (std::size_t i = 0; i < size_in_words; i++) {
        ASSERT_EQ(args[i], i);
    }
}

TEST(CclCommandArgGenerator, UnpackFullTensorArg) {
    constexpr std::size_t size_in_words = full_tensor_command_arg_t::size_in_words();
    ASSERT_EQ(size_in_words, 17);
    std::array<uint32_t, full_tensor_command_arg_t::size_in_words()> args;
    std::iota(args.begin(), args.end(), 0);

    full_tensor_command_arg_t::field_type test_tensor = {
        {std::numeric_limits<uint32_t>::max(),std::numeric_limits<uint32_t>::max(),std::numeric_limits<uint32_t>::max(),std::numeric_limits<uint32_t>::max()},
        {std::numeric_limits<uint32_t>::max(),std::numeric_limits<uint32_t>::max(),std::numeric_limits<uint32_t>::max(),std::numeric_limits<uint32_t>::max()},
        {std::numeric_limits<uint32_t>::max(),std::numeric_limits<uint32_t>::max(),std::numeric_limits<uint32_t>::max(),std::numeric_limits<uint32_t>::max()},
        {std::numeric_limits<uint32_t>::max(),std::numeric_limits<uint32_t>::max(),std::numeric_limits<uint32_t>::max(),std::numeric_limits<uint32_t>::max()},
        std::numeric_limits<uint32_t>::max()
    };
    full_tensor_command_arg_t::unpack(args.data(), test_tensor);
    ASSERT_EQ(test_tensor.tensor_shape.w, 0);
    ASSERT_EQ(test_tensor.tensor_shape.z, 1);
    ASSERT_EQ(test_tensor.tensor_shape.y, 2);
    ASSERT_EQ(test_tensor.tensor_shape.x, 3);
    ASSERT_EQ(test_tensor.tensor_slice_shape.w, 4);
    ASSERT_EQ(test_tensor.tensor_slice_shape.z, 5);
    ASSERT_EQ(test_tensor.tensor_slice_shape.y, 6);
    ASSERT_EQ(test_tensor.tensor_slice_shape.x, 7);
    ASSERT_EQ(test_tensor.tensor_slice_offset.w, 8);
    ASSERT_EQ(test_tensor.tensor_slice_offset.z, 9);
    ASSERT_EQ(test_tensor.tensor_slice_offset.y, 10);
    ASSERT_EQ(test_tensor.tensor_slice_offset.x, 11);
    ASSERT_EQ(test_tensor.worker_start_offset_in_slice.w, 12);
    ASSERT_EQ(test_tensor.worker_start_offset_in_slice.z, 13);
    ASSERT_EQ(test_tensor.worker_start_offset_in_slice.y, 14);
    ASSERT_EQ(test_tensor.worker_start_offset_in_slice.x, 15);
    ASSERT_EQ(test_tensor.worker_pages_per_slice, 16);
}
