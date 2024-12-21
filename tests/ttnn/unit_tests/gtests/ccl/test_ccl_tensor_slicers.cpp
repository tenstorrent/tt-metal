// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_command.hpp"

#include "gtest/gtest.h"

#include <cstddef>

TEST(
    CclTensorSlicer_SliceWorkerSplitting,
    page_based_1worker_TensorShape_1_1_4_1__SliceShape_1_1_1_1__SliceOffset_0_0_3_0__Workers1) {
    const auto worker_slices = ttnn::ccl::cmd::builder::split_tensor_slice_across_workers_wrapped_page_aligned(
        ttnn::ccl::v2::TensorSlice{
            {1, 1, 4, 1},  // tensor_shape
            {1, 1, 1, 1},  // tensor slice shape
            {0, 0, 3, 0},  // tensor slice offset
            {1, 1, 1, 1},
            {0, 0, 0, 0}},
        1);

    ASSERT_EQ(worker_slices.size(), 1);
    ASSERT_EQ(worker_slices[0].tensor_slice_shape.w, 1);
    ASSERT_EQ(worker_slices[0].tensor_slice_shape.z, 1);
    ASSERT_EQ(worker_slices[0].tensor_slice_shape.y, 1);
    ASSERT_EQ(worker_slices[0].tensor_slice_shape.x, 1);

    ASSERT_EQ(worker_slices[0].tensor_slice_offset.w, 0);
    ASSERT_EQ(worker_slices[0].tensor_slice_offset.z, 0);
    ASSERT_EQ(worker_slices[0].tensor_slice_offset.y, 3);
    ASSERT_EQ(worker_slices[0].tensor_slice_offset.x, 0);

    ASSERT_EQ(worker_slices[0].worker_slice_shape.w, 1);
    ASSERT_EQ(worker_slices[0].worker_slice_shape.z, 1);
    ASSERT_EQ(worker_slices[0].worker_slice_shape.y, 1);
    ASSERT_EQ(worker_slices[0].worker_slice_shape.x, 1);

    ASSERT_EQ(worker_slices[0].worker_slice_offset.w, 0);
    ASSERT_EQ(worker_slices[0].worker_slice_offset.z, 0);
    ASSERT_EQ(worker_slices[0].worker_slice_offset.y, 0);
    ASSERT_EQ(worker_slices[0].worker_slice_offset.x, 0);
}

TEST(
    CclTensorSlicer_SliceWorkerSplitting,
    page_based_1worker_TensorShape_1_1_4_1__SliceShape_1_1_1_1__SliceOffset_0_0_0_0__Workers1) {
    const auto worker_slices = ttnn::ccl::cmd::builder::split_tensor_slice_across_workers_wrapped_page_aligned(
        ttnn::ccl::v2::TensorSlice{
            {1, 1, 4, 1},  // tensor_shape
            {1, 1, 1, 1},  // tensor slice shape
            {0, 0, 0, 0},  // tensor slice offset
            {1, 1, 4, 1},
            {0, 0, 0, 0}},
        1);

    ASSERT_EQ(worker_slices.size(), 1);
    ASSERT_EQ(worker_slices[0].tensor_slice_shape.w, 1);
    ASSERT_EQ(worker_slices[0].tensor_slice_shape.z, 1);
    ASSERT_EQ(worker_slices[0].tensor_slice_shape.y, 1);
    ASSERT_EQ(worker_slices[0].tensor_slice_shape.x, 1);

    ASSERT_EQ(worker_slices[0].tensor_slice_offset.w, 0);
    ASSERT_EQ(worker_slices[0].tensor_slice_offset.z, 0);
    ASSERT_EQ(worker_slices[0].tensor_slice_offset.y, 0);
    ASSERT_EQ(worker_slices[0].tensor_slice_offset.x, 0);

    ASSERT_EQ(worker_slices[0].worker_slice_shape.w, 1);
    ASSERT_EQ(worker_slices[0].worker_slice_shape.z, 1);
    ASSERT_EQ(worker_slices[0].worker_slice_shape.y, 1);
    ASSERT_EQ(worker_slices[0].worker_slice_shape.x, 1);

    ASSERT_EQ(worker_slices[0].worker_slice_offset.w, 0);
    ASSERT_EQ(worker_slices[0].worker_slice_offset.z, 0);
    ASSERT_EQ(worker_slices[0].worker_slice_offset.y, 0);
    ASSERT_EQ(worker_slices[0].worker_slice_offset.x, 0);
}
