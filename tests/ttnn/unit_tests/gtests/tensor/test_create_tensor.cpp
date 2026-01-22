// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/move/utility_core.hpp>
#include <fmt/base.h>
#include <enchantum/enchantum.hpp>
#include <cstdint>
#include <tt-logger/tt-logger.hpp>
#include <initializer_list>
#include <memory>
#include <optional>
#include <ostream>
#include <string_view>
#include <tuple>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include "common_tensor_test_utils.hpp"
#include "gtest/gtest.h"
#include <tt-metalium/shape.hpp>
#include "ttnn/async_runtime.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace tt::tt_metal {
class IDevice;
}  // namespace tt::tt_metal

namespace {

void run_create_tensor_test(tt::tt_metal::distributed::MeshDevice* device, const ttnn::Shape& input_shape) {
    MemoryConfig mem_cfg = MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};

    const ttnn::QueueId io_cq = ttnn::QueueId(0);
    constexpr DataType dtype = DataType::BFLOAT16;
    constexpr uint32_t datum_size_bytes = 2;

    auto input_buf_size_datums = input_shape.volume();

    auto host_data = std::shared_ptr<uint16_t[]>(new uint16_t[input_buf_size_datums]);
    auto readback_data = std::shared_ptr<uint16_t[]>(new uint16_t[input_buf_size_datums]);

    for (int i = 0; i < input_buf_size_datums; i++) {
        host_data[i] = 1;
    }

    TensorSpec tensor_spec(input_shape, TensorLayout(dtype, PageConfig(Layout::TILE), mem_cfg));
    ASSERT_EQ(input_buf_size_datums * datum_size_bytes, tensor_spec.compute_packed_buffer_size_bytes());
    auto input_buffer = tt::tt_metal::tensor_impl::allocate_device_buffer(device, tensor_spec);

    auto input_storage = tt::tt_metal::DeviceStorage{input_buffer, {tt::tt_metal::distributed::MeshCoordinate{0, 0}}};

    Tensor input_tensor = Tensor(input_storage, tensor_spec, TensorTopology{});

    ttnn::write_buffer(io_cq, input_tensor, {host_data});

    ttnn::read_buffer(io_cq, input_tensor, {readback_data});

    for (int i = 0; i < input_buf_size_datums; i++) {
        EXPECT_EQ(host_data[i], readback_data[i]);
    }

    input_tensor.deallocate();
}

struct CreateTensorParams {
    ttnn::Shape shape;
};

}  // namespace

class CreateTensorTest : public ttnn::TTNNFixtureWithSuiteDevice<CreateTensorTest>,
                         public ::testing::WithParamInterface<CreateTensorParams> {};

TEST_P(CreateTensorTest, Tile) {
    const CreateTensorParams& params = GetParam();
    run_create_tensor_test(device_, params.shape);
}

INSTANTIATE_TEST_SUITE_P(
    CreateTensorTestWithShape,
    CreateTensorTest,
    ::testing::Values(
        CreateTensorParams{.shape = ttnn::Shape({1, 1, 32, 32})},
        CreateTensorParams{.shape = ttnn::Shape({2, 1, 32, 32})},
        CreateTensorParams{.shape = ttnn::Shape({0, 0, 0, 0})},
        CreateTensorParams{.shape = ttnn::Shape({0, 1, 32, 32})},
        CreateTensorParams{.shape = ttnn::Shape({0})}));

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::DataType& value) {
    os << enchantum::to_string(value);
    return os;
}

using CombinationInputParams =
    std::tuple<ttnn::Shape, tt::tt_metal::DataType, tt::tt_metal::Layout, tt::tt_metal::MemoryConfig>;
class EmptyTensorTest : public ttnn::TTNNFixtureWithSuiteDevice<EmptyTensorTest>,
                        public ::testing::WithParamInterface<CombinationInputParams> {};

TEST_P(EmptyTensorTest, Combinations) {
    auto params = GetParam();
    auto shape = std::get<0>(params);
    auto dtype = std::get<1>(params);
    auto layout = std::get<2>(params);
    auto memory_config = std::get<3>(params);
    log_info(
        tt::LogTest,
        "Running test with shape={}, dtype={}, layout={}, memory_config={}",
        shape,
        dtype,
        layout,
        memory_config);

    if (layout == tt::tt_metal::Layout::ROW_MAJOR && dtype == tt::tt_metal::DataType::BFLOAT8_B) {
        GTEST_SKIP() << "Skipping test with ROW_MAJOR layout and BFLOAT8_B dtype!";
    }

    auto tensor_layout = tt::tt_metal::TensorLayout::fromPaddedShape(
        dtype, PageConfig(layout), memory_config, /* logical */ shape, /* padded */ shape);

    auto tensor = tt::tt_metal::create_device_tensor(
        ttnn::TensorSpec(shape, tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(layout), memory_config)),
        device_);
    EXPECT_EQ(tensor.logical_shape(), shape);

    test_utils::test_tensor_on_device(shape, tensor_layout, device_);
}

INSTANTIATE_TEST_SUITE_P(
    EmptyTensorTestWithShape,
    EmptyTensorTest,
    ::testing::Combine(
        ::testing::Values(
            ttnn::Shape({}),
            ttnn::Shape({0}),
            ttnn::Shape({1}),
            ttnn::Shape({1, 2}),
            ttnn::Shape({1, 2, 3}),
            ttnn::Shape({1, 2, 3, 4}),
            // ttnn::Shape({0, 0, 0, 0}), fails with width sharded case
            ttnn::Shape({1, 1, 1, 1}),
            // ttnn::Shape({0, 1, 32, 32}), fails with width sharded case
            ttnn::Shape({1, 1, 32, 32}),
            ttnn::Shape({2, 1, 32, 32}),
            ttnn::Shape({64, 1, 256, 1}),
            ttnn::Shape({1, 1, 21120, 16}),
            ttnn::Shape({1, 2, 3, 4, 5})),

        ::testing::Values(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::DataType::FLOAT32,
            tt::tt_metal::DataType::BFLOAT8_B),

        ::testing::Values(tt::tt_metal::Layout::TILE, tt::tt_metal::Layout::ROW_MAJOR),

        ::testing::Values(
            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::L1},

            tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM}

            // tt::tt_metal::MemoryConfig{
            //     .memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
            //     .buffer_type = tt::tt_metal::BufferType::L1,
            //     .shard_spec = ShardSpec{
            //         CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{4, 1}}}},
            //         {32, 64},
            //         tt::tt_metal::ShardOrientation::ROW_MAJOR,
            //         false}
            // },
            // tt::tt_metal::MemoryConfig{
            //     .memory_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
            //     .buffer_type = tt::tt_metal::BufferType::DRAM,
            //     .shard_spec = ShardSpec{
            //         CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{4, 1}}}},
            //         {32, 64},
            //         tt::tt_metal::ShardOrientation::ROW_MAJOR,
            //         false}
            // },

            // ttnn::MemoryConfig{
            //     .memory_layout = tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
            //     .buffer_type = tt::tt_metal::BufferType::L1,
            //     .shard_spec = ShardSpec{
            //         CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{1, 4}}}},
            //         {32, 64},
            //         tt::tt_metal::ShardOrientation::ROW_MAJOR,
            //         false}
            // },
            // tt::tt_metal::MemoryConfig{
            //     .memory_layout = tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
            //     .buffer_type = tt::tt_metal::BufferType::DRAM,
            //     .shard_spec = ShardSpec{
            //         CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{1, 4}}}},
            //         {32, 64},
            //         tt::tt_metal::ShardOrientation::ROW_MAJOR,
            //         false}
            // },

            // tt::tt_metal::MemoryConfig{
            //     .memory_layout = tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
            //     .buffer_type = tt::tt_metal::BufferType::L1,
            //     .shard_spec = ShardSpec{
            //         CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{4, 4}}}},
            //         {64, 64},
            //         tt::tt_metal::ShardOrientation::ROW_MAJOR,
            //         false}
            // }
            // tt::tt_metal::MemoryConfig{
            //     .memory_layout = tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
            //     .buffer_type = tt::tt_metal::BufferType::DRAM,
            //     .shard_spec = ShardSpec{
            //         CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{4, 4}}}},
            //         {64, 64},
            //         tt::tt_metal::ShardOrientation::ROW_MAJOR,
            //         false}
            // }
            )));
