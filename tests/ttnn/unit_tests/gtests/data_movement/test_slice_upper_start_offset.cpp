// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdint>
#include <string>
#include <vector>

#include <tt-metalium/constants.hpp>
#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::data_movement::test {
namespace {

// Mirror of the fixed forward Horner loop in
// reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp
uint32_t kernel_forward_upper_tile_offset(const ttnn::Shape& shape, const ttnn::Shape& start) {
    uint32_t upper_dims_offset = 0;
    for (uint32_t i = 0; i + 2 < shape.rank(); ++i) {
        upper_dims_offset = upper_dims_offset * shape[i] + start[i];
    }
    const uint32_t multiplier = (shape[-2] / tt::constants::TILE_HEIGHT) * (shape[-1] / tt::constants::TILE_WIDTH);
    return upper_dims_offset * multiplier;
}

// Pre-fix kernel loop (reverse Horner) — diverges from host for rank >= 4.
uint32_t kernel_reverse_upper_tile_offset(const ttnn::Shape& shape, const ttnn::Shape& start) {
    uint32_t upper_dims_offset = 0;
    for (int32_t i = static_cast<int32_t>(shape.rank()) - 3; i >= 0; i--) {
        upper_dims_offset = upper_dims_offset * shape[i] + start[i];
    }
    const uint32_t multiplier = (shape[-2] / tt::constants::TILE_HEIGHT) * (shape[-1] / tt::constants::TILE_WIDTH);
    return upper_dims_offset * multiplier;
}

std::string shape_start_trace_label(const ttnn::Shape& shape, const ttnn::Shape& slice_start) {
    std::string label = "shape=[";
    for (uint32_t i = 0; i < shape.rank(); ++i) {
        if (i > 0) {
            label += ", ";
        }
        label += std::to_string(shape[i]);
    }
    label += "], start=[";
    for (uint32_t i = 0; i < slice_start.rank(); ++i) {
        if (i > 0) {
            label += ", ";
        }
        label += std::to_string(slice_start[i]);
    }
    label += "]";
    return label;
}

void expect_kernel_and_host_agree(const ttnn::Shape& shape, const ttnn::Shape& slice_start) {
    SCOPED_TRACE(shape_start_trace_label(shape, slice_start));

    // Shapes must be tile-aligned (kernel uses padded_shape dimensions).
    ASSERT_EQ(shape[-2] % tt::constants::TILE_HEIGHT, 0u);
    ASSERT_EQ(shape[-1] % tt::constants::TILE_WIDTH, 0u);

    const uint32_t host_offset = get_upper_start_offset(shape, Layout::TILE, slice_start);
    const uint32_t kernel_offset = kernel_forward_upper_tile_offset(shape, slice_start);
    EXPECT_EQ(kernel_offset, host_offset);
}

struct UpperStartOffsetCase {
    std::vector<uint32_t> shape;
    std::vector<uint32_t> start;
};

class KernelAndHostUpperStartOffsetFixture : public testing::TestWithParam<UpperStartOffsetCase> {};

TEST_P(KernelAndHostUpperStartOffsetFixture, KernelAndHostUpperStartOffsetAgree) {
    const auto& param = GetParam();
    const ttnn::Shape shape(param.shape);
    const ttnn::Shape slice_start(param.start);

    ASSERT_NO_FATAL_FAILURE(expect_kernel_and_host_agree(shape, slice_start));

    const uint32_t host_offset = get_upper_start_offset(shape, Layout::TILE, slice_start);
    const uint32_t reverse_offset = kernel_reverse_upper_tile_offset(shape, slice_start);
    EXPECT_NE(reverse_offset, host_offset);
}

INSTANTIATE_TEST_SUITE_P(
    SliceHornerLoop,
    KernelAndHostUpperStartOffsetFixture,
    testing::Values(
        UpperStartOffsetCase{{2, 3, 64, 64}, {1, 0, 0, 0}},
        UpperStartOffsetCase{{4, 8, 64, 64}, {1, 3, 0, 0}},
        UpperStartOffsetCase{{2, 5, 32, 32}, {1, 1, 0, 0}},
        UpperStartOffsetCase{{3, 2, 4, 64, 64}, {1, 1, 1, 0, 0}}),
    [](const testing::TestParamInfo<UpperStartOffsetCase>& info) { return "case" + std::to_string(info.index); });

// Rank 3 has a single upper dim; forward and reverse Horner loops coincide.
TEST(SliceUpperStartOffset, Rank3ForwardAndReverseHornerAgreeWithHost) {
    const ttnn::Shape shape({3, 64, 64});
    const ttnn::Shape slice_start({1, 0, 0});

    ASSERT_NO_FATAL_FAILURE(expect_kernel_and_host_agree(shape, slice_start));
    EXPECT_EQ(
        kernel_forward_upper_tile_offset(shape, slice_start), kernel_reverse_upper_tile_offset(shape, slice_start));
}

TEST(SliceUpperStartOffset, MultiUpperDimKernelAndHostAgreement) {
    const std::vector<std::vector<uint32_t>> shapes = {
        {2, 3, 64, 64},
        {4, 8, 64, 64},
        {2, 5, 32, 32},
        {6, 4, 32, 32},
        {3, 2, 4, 64, 64},
        {6, 3, 2, 32, 32},
    };

    for (const auto& shape_vec : shapes) {
        const ttnn::Shape shape(shape_vec);
        std::vector<uint32_t> start_vec(shape_vec.size(), 0);

        for (uint32_t d = 0; d + 2 < shape.rank(); ++d) {
            if (shape[d] <= 1) {
                continue;
            }
            start_vec[d] = shape[d] / 2;
            ASSERT_NO_FATAL_FAILURE(expect_kernel_and_host_agree(shape, ttnn::Shape(start_vec)));
        }
    }
}

}  // namespace
}  // namespace ttnn::operations::data_movement::test
