// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstddef>
#include <vector>

#include "gtest/gtest.h"
#include "ttnn/cpp/ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_command.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/common/types/ccl_types.hpp"

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

static size_t get_flat_index_from_shape(
    const ttnn::ccl::Shape4D<uint32_t>& shape, const ttnn::ccl::Shape4D<uint32_t>& index) {
    std::size_t offset = index.x;
    std::size_t inner_volume = shape.x;
    offset += index.y * inner_volume;
    inner_volume *= shape.y;
    offset += index.z * inner_volume;
    inner_volume *= shape.z;
    offset += index.w * inner_volume;
    return offset;
}

TEST(
    TensorIterationSweep,
    advance_worker_global_page__Shape_1_4_4_72__SliceShape_1_4_1_72__SliceOffset_0_0_3_0__WorkerStartPage_0__Stride_1) {
    uint32_t stride = 1;
    ttnn::ccl::Shape4D<uint32_t> tensor_shape{1, 4, 4, 72};
    ttnn::ccl::Shape4D<uint32_t> tensor_slice_shape{1, 4, 1, 72};
    const ttnn::ccl::Shape4D<uint32_t> tensor_slice_offset_base{0, 0, 3, 0};
    ttnn::ccl::Shape4D<uint32_t> tensor_slice_offset_current{0, 0, 3, 0};
    ttnn::ccl::Shape4D<uint32_t> start_offset_worker_slice{0, 0, 0, 0};
    size_t worker_slice_volume = tensor_slice_shape.volume();
    uint32_t curr_page_idx =
        get_flat_index_from_shape(tensor_shape, tensor_slice_offset_base + start_offset_worker_slice);
    uint32_t offset_into_worker_slice = 0;

    tensor_slice_offset_current.w = tensor_slice_offset_base.w;
    for (size_t w = 0; w < tensor_slice_shape.w; w++) {
        bool last_w = w == tensor_slice_shape.w - 1;
        tensor_slice_offset_current.z = tensor_slice_offset_base.z;
        for (size_t z = 0; z < tensor_slice_shape.z; z++) {
            bool last_z = z == tensor_slice_shape.z - 1;
            tensor_slice_offset_current.y = tensor_slice_offset_base.y;
            for (size_t y = 0; y < tensor_slice_shape.y; y++) {
                bool last_y = y == tensor_slice_shape.y - 1;
                tensor_slice_offset_current.x = tensor_slice_offset_base.x;
                for (size_t x = 0; x < tensor_slice_shape.x; x++) {
                    bool last_x = x == tensor_slice_shape.x - 1;
                    bool end_of_worker_slice = ttnn::ccl::v2::advance_worker_global_page(
                        curr_page_idx,
                        offset_into_worker_slice,   // local to the worker chunk
                        start_offset_worker_slice,  // local to the tensor slice

                        worker_slice_volume,  // worker chunk shape
                        tensor_slice_shape,   // tensor slice shape (per device)
                        tensor_slice_offset_base,

                        tensor_shape,  // full tensor shape

                        stride);
                    if (tensor_slice_offset_current.x == (tensor_slice_offset_base.x + tensor_slice_shape.x - 1)) {
                        if (tensor_slice_offset_current.y == (tensor_slice_offset_base.y + tensor_slice_shape.y - 1)) {
                            if (tensor_slice_offset_current.z ==
                                (tensor_slice_offset_base.z + tensor_slice_shape.z - 1)) {
                                tensor_slice_offset_current.w =
                                    (tensor_slice_offset_current.w + 1) % tensor_slice_shape.w;
                                tensor_slice_offset_current.z = tensor_slice_offset_base.z;
                            } else {
                                tensor_slice_offset_current.z = tensor_slice_offset_current.z + 1;
                            }
                            tensor_slice_offset_current.y = tensor_slice_offset_base.y;
                        } else {
                            tensor_slice_offset_current.y = tensor_slice_offset_current.y + 1;
                        }
                        tensor_slice_offset_current.x = tensor_slice_offset_base.x;
                    } else {
                        tensor_slice_offset_current.x = tensor_slice_offset_current.x + 1;
                    }
                    bool last_page = last_w && last_z && last_y && last_x;
                    ASSERT_TRUE(
                        last_page ||
                        (curr_page_idx == get_flat_index_from_shape(tensor_shape, tensor_slice_offset_current)));
                }
            }
        }
    }
}

TEST(
    TensorIteration,
    advance_worker_global_page__Shape_1_4_4_72__SliceShape_1_4_1_72__OffsetIntoWorkerSlice_71__CurrPageId_287_last_page_on_plane__Stride_1) {
    uint32_t stride = 1;
    uint32_t curr_page_idx = 287;
    uint32_t offset_into_worker_slice = 71;
    ttnn::ccl::Shape4D<uint32_t> tensor_shape{1, 4, 4, 72};
    ttnn::ccl::Shape4D<uint32_t> tensor_slice_shape{1, 4, 1, 72};
    ttnn::ccl::Shape4D<uint32_t> tensor_slice_offset{0, 0, 3, 0};
    ttnn::ccl::Shape4D<uint32_t> start_offset_worker_slice{0, 0, 0, 0};
    ttnn::ccl::Shape4D<uint32_t> worker_slice_shape{1, 1, 1, 288};

    auto old_page_id = curr_page_idx;
    bool end_of_worker_slice = ttnn::ccl::v2::advance_worker_global_page(
        curr_page_idx,
        offset_into_worker_slice,   // local to the worker chunk
        start_offset_worker_slice,  // local to the tensor slice

        worker_slice_shape.volume(),  // worker chunk shape
        tensor_slice_shape,           // tensor slice shape (per device)
        tensor_slice_offset,

        tensor_shape,  // full tensor shape

        stride);
    ASSERT_EQ(curr_page_idx, old_page_id + 1 + 72 * 3);
    ASSERT_EQ(offset_into_worker_slice, 72);
}
