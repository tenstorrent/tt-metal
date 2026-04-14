// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <array>
#include <memory>
#include <optional>

#include <tt_stl/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/shape.hpp>
#include "ttnn/device.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::binary::test {

struct SumTensorParameter {
    int h;
    int w;
};

class SumTensorLastDimFixture : public TTNNFixtureWithSuiteDevice<SumTensorLastDimFixture>,
                                public testing::WithParamInterface<SumTensorParameter> {};

TEST_P(SumTensorLastDimFixture, SumTensorCorrectly) {
    auto param = GetParam();
    auto& device = *device_;
    std::array<uint32_t, 2> dimensions = {param.h, param.w};
    ttnn::Shape shape(dimensions);
    std::array<uint32_t, 2> reduced_dimensions = {param.h, 1};
    ttnn::Shape reduced_shape(reduced_dimensions);
    constexpr int dim = -1;
    {
        const auto input_tensor = ttnn::ones(shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
        const auto output_tensor = ttnn::sum(input_tensor, dim, true);
        TT_FATAL(
            output_tensor.logical_shape() == reduced_shape,
            "Shapes are not equal output tensor shape {} vs reduced shape {}",
            output_tensor.logical_shape(),
            reduced_shape);
        auto output_vector = output_tensor.to_vector<bfloat16>();
        float expected = (float)param.w;
        for (int i = 0; i < param.h; i++) {
            float value = output_vector[i];
            TT_FATAL(value == expected, "{} != {} @ {}", value, expected, i);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    SumTensorLastDimTests,
    SumTensorLastDimFixture,
    ::testing::Values(SumTensorParameter{3100, 63}, SumTensorParameter{3200, 64}));

class SumTensorFirstDimFixture : public TTNNFixtureWithSuiteDevice<SumTensorFirstDimFixture>,
                                 public testing::WithParamInterface<SumTensorParameter> {};

TEST_P(SumTensorFirstDimFixture, SumTensorCorrectly) {
    auto param = GetParam();
    auto& device = *device_;
    std::array<uint32_t, 2> dimensions = {param.h, param.w};
    ttnn::Shape shape(dimensions);
    std::array<uint32_t, 2> reduced_dimensions = {1, param.w};
    ttnn::Shape reduced_shape(reduced_dimensions);
    constexpr int dim = -2;
    {
        const auto input_tensor = ttnn::ones(shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
        const auto output_tensor = ttnn::sum(input_tensor, dim, true);
        TT_FATAL(
            output_tensor.logical_shape() == reduced_shape,
            "Shapes are not equal output tensor shape {} vs reduced shape {}",
            output_tensor.logical_shape(),
            reduced_shape);
        auto output_vector = output_tensor.to_vector<bfloat16>();
        float expected = (float)param.h;
        for (int i = 0; i < param.w; i++) {
            float value = output_vector[i];
            TT_FATAL(value == expected, "{} != {} @ {}", value, expected, i);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    SumTensorFirstDimTests,
    SumTensorFirstDimFixture,
    ::testing::Values(SumTensorParameter{63, 3100}, SumTensorParameter{64, 3200}));

class SumTensorBothDimsFixture : public TTNNFixtureWithSuiteDevice<SumTensorBothDimsFixture>,
                                 public testing::WithParamInterface<SumTensorParameter> {};

TEST_P(SumTensorBothDimsFixture, SumTensorCorrectly) {
    auto param = GetParam();
    auto& device = *device_;
    std::array<uint32_t, 2> dimensions = {param.h, param.w};
    ttnn::Shape shape(dimensions);
    SmallVector<int> dim = {0, 1};

    {
        const auto input_tensor = ttnn::ones(shape, DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
        const auto output_tensor = ttnn::sum(input_tensor, dim, false);
        auto output_vector = output_tensor.to_vector<bfloat16>();
        float expected = (float)(param.h * param.w);
        TT_FATAL(
            output_vector.size() == 1, "Result should only have a single element instead of {}", output_vector.size());
        float value = output_vector[0];
        TT_FATAL(value == expected, "{} != {}", value, expected);
    }
}

INSTANTIATE_TEST_SUITE_P(
    SumTensorBothDimsTests,
    SumTensorBothDimsFixture,
    ::testing::Values(SumTensorParameter{30, 30}, SumTensorParameter{64, 64}));

struct MinMaxTensorParameter {
    int h;
    int w;
    int offset;
};

class MinMaxTensorLastDimFixture : public TTNNFixtureWithSuiteDevice<MinMaxTensorLastDimFixture>,
                                   public testing::WithParamInterface<MinMaxTensorParameter> {};

TEST_P(MinMaxTensorLastDimFixture, MinMaxTensorCorrectly) {
    auto param = GetParam();
    auto& device = *device_;
    std::array<uint32_t, 4> reduced_dimensions = {1, 1, param.h, 1};
    ttnn::Shape reduced_shape(reduced_dimensions);
    int dim = -1;

    {
        const ttnn::Shape tensor_shape{1, 1, param.h, param.w};
        const MemoryConfig mem_cfg = MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
        const TensorLayout tensor_layout(DataType::BFLOAT16, PageConfig(Layout::TILE), mem_cfg);
        const TensorSpec tensor_spec(tensor_shape, tensor_layout);
        std::vector<float> host_data(tensor_shape.volume());
        for (int i = 0; i < param.h; i++) {
            for (int j = 0; j < param.w; j++) {
                int index = param.w * i + j;
                host_data[index] = param.offset + j;
            }
        }

        const ttnn::QueueId io_cq = ttnn::QueueId(0);
        const Tensor host_tensor = Tensor::from_vector(host_data, tensor_spec);
        const Tensor input_tensor = host_tensor.to_device(&device, mem_cfg, io_cq);
        const auto output_tensor_max = ttnn::max(input_tensor, dim, true);
        const auto output_tensor_min = ttnn::min(input_tensor, dim, true);
        TT_FATAL(
            output_tensor_max.logical_shape() == reduced_shape,
            "Shapes are not equal output tensor shape {} vs reduced shape {}",
            output_tensor_max.logical_shape(),
            reduced_shape);
        TT_FATAL(
            output_tensor_min.logical_shape() == reduced_shape,
            "Shapes are not equal output tensor shape {} vs reduced shape {}",
            output_tensor_max.logical_shape(),
            reduced_shape);
        auto output_vector_max = output_tensor_max.to_vector<bfloat16>();
        auto output_vector_min = output_tensor_min.to_vector<bfloat16>();
        float expected_max = (float)(param.offset + param.w - 1.0);
        float expected_min = (float)param.offset;
        for (int i = 0; i < param.h; i++) {
            float max = output_vector_max[i];
            float min = output_vector_min[i];
            TT_FATAL(max == expected_max, "{} != {} @ {}", max, expected_max, i);
            TT_FATAL(min == expected_min, "{} != {} @ {}", min, expected_min, i);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    MinMaxTensorLastDimTests,
    MinMaxTensorLastDimFixture,
    ::testing::Values(
        MinMaxTensorParameter{3100, 63, 4},
        MinMaxTensorParameter{3200, 64, 4},
        MinMaxTensorParameter{3100, 63, -128},
        MinMaxTensorParameter{3200, 64, -128}));

class MinMaxTensorFirstDimFixture : public TTNNFixtureWithSuiteDevice<MinMaxTensorFirstDimFixture>,
                                    public testing::WithParamInterface<MinMaxTensorParameter> {};

TEST_P(MinMaxTensorFirstDimFixture, MinMaxTensorCorrectly) {
    auto param = GetParam();
    auto& device = *device_;
    std::array<uint32_t, 4> reduced_dimensions = {1, 1, 1, param.w};
    ttnn::Shape reduced_shape(reduced_dimensions);
    int dim = -2;
    {
        const ttnn::Shape tensor_shape{1, 1, param.h, param.w};
        const MemoryConfig mem_cfg = MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
        const TensorLayout tensor_layout(DataType::BFLOAT16, PageConfig(Layout::TILE), mem_cfg);
        const TensorSpec tensor_spec(tensor_shape, tensor_layout);
        std::vector<float> host_data(tensor_shape.volume());
        for (int i = 0; i < param.h; i++) {
            for (int j = 0; j < param.w; j++) {
                int index = param.w * i + j;
                host_data[index] = param.offset + i;
            }
        }

        const ttnn::QueueId io_cq = ttnn::QueueId(0);
        const Tensor host_tensor = Tensor::from_vector(host_data, tensor_spec);
        const Tensor input_tensor = host_tensor.to_device(&device, mem_cfg, io_cq);
        const auto output_tensor_max = ttnn::max(input_tensor, dim, true);
        const auto output_tensor_min = ttnn::min(input_tensor, dim, true);
        TT_FATAL(
            output_tensor_max.logical_shape() == reduced_shape,
            "Shapes are not equal output tensor shape {} vs reduced shape {}",
            output_tensor_max.logical_shape(),
            reduced_shape);
        TT_FATAL(
            output_tensor_min.logical_shape() == reduced_shape,
            "Shapes are not equal output tensor shape {} vs reduced shape {}",
            output_tensor_max.logical_shape(),
            reduced_shape);
        auto output_vector_max = output_tensor_max.to_vector<bfloat16>();
        auto output_vector_min = output_tensor_min.to_vector<bfloat16>();
        for (int i = 0; i < param.w; i++) {
            float expected_max = (float)(param.offset + param.h - 1.0);
            float expected_min = (float)param.offset;
            float max = output_vector_max[i];
            float min = output_vector_min[i];
            TT_FATAL(max == expected_max, "{} != {} @ {}", max, expected_max, i);
            TT_FATAL(min == expected_min, "{} != {} @ {}", min, expected_min, i);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    MinMaxTensorFirstDimTests,
    MinMaxTensorFirstDimFixture,
    ::testing::Values(
        MinMaxTensorParameter{63, 3100, 4},
        MinMaxTensorParameter{64, 3200, 4},
        MinMaxTensorParameter{63, 3100, -128},
        MinMaxTensorParameter{64, 3200, -128}));

class MinMaxTensorBothDimsFixture : public TTNNFixtureWithSuiteDevice<MinMaxTensorBothDimsFixture>,
                                    public testing::WithParamInterface<MinMaxTensorParameter> {};

TEST_P(MinMaxTensorBothDimsFixture, MinMaxTensorCorrectly) {
    auto param = GetParam();
    auto& device = *device_;
    SmallVector<int> dim = {-2, -1};
    {
        const ttnn::Shape tensor_shape{1, 1, param.h, param.w};
        const MemoryConfig mem_cfg = MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
        const TensorLayout tensor_layout(DataType::BFLOAT16, PageConfig(Layout::TILE), mem_cfg);
        const TensorSpec tensor_spec(tensor_shape, tensor_layout);
        std::vector<float> host_data(tensor_shape.volume());
        for (int i = 0; i < param.h; i++) {
            for (int j = 0; j < param.w; j++) {
                int index = param.w * i + j;
                host_data[index] = param.offset + index;
            }
        }

        const ttnn::QueueId io_cq = ttnn::QueueId(0);
        const Tensor host_tensor = Tensor::from_vector(host_data, tensor_spec);
        const Tensor input_tensor = host_tensor.to_device(&device, mem_cfg, io_cq);
        const auto output_tensor_max = ttnn::max(input_tensor, dim, true);
        const auto output_tensor_min = ttnn::min(input_tensor, dim, true);
        auto output_vector_max = output_tensor_max.to_vector<bfloat16>();
        auto output_vector_min = output_tensor_min.to_vector<bfloat16>();
        TT_FATAL(
            output_vector_max.size() == 1,
            "Result should only have a single element instead of {}",
            output_vector_max.size());
        TT_FATAL(
            output_vector_min.size() == 1,
            "Result should only have a single element instead of {}",
            output_vector_min.size());
        float expected_max = (float)(param.offset + param.h * param.w - 1.0);
        float expected_min = (float)param.offset;
        float max = output_vector_max[0];
        float min = output_vector_min[0];
        TT_FATAL(max == expected_max, "{} != {}", max, expected_max);
        TT_FATAL(min == expected_min, "{} != {}", min, expected_min);
    }
}

INSTANTIATE_TEST_SUITE_P(
    MinMaxTensorBothDimsTests,
    MinMaxTensorBothDimsFixture,
    ::testing::Values(MinMaxTensorParameter{64, 64, 1}, MinMaxTensorParameter{30, 30, -1004}));

}  // namespace ttnn::operations::binary::test
