// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "tt_metal/common/bfloat16.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/async_runtime.hpp"
#include "ttnn/operations/numpy/functions.hpp"
#include "tt_metal/common/logger.hpp"

#include "common_tensor_test_utils.hpp"

#include "ttnn_test_fixtures.hpp"

namespace {

void run_create_tensor_test(tt::tt_metal::Device* device, ttnn::SimpleShape input_shape) {
    MemoryConfig mem_cfg = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        .buffer_type = BufferType::DRAM,
        .shard_spec = std::nullopt};

    const uint32_t io_cq = 0;
    constexpr DataType dtype = DataType::BFLOAT16;
    constexpr uint32_t datum_size_bytes = 2;

    auto input_buf_size_datums = input_shape.volume();

    auto host_data = std::shared_ptr<uint16_t[]>(new uint16_t[input_buf_size_datums]);
    auto readback_data = std::shared_ptr<uint16_t[]>(new uint16_t[input_buf_size_datums]);

    for (int i = 0; i < input_buf_size_datums; i++) {
        host_data[i] = 1;
    }

    tt::tt_metal::TensorLayout tensor_layout(dtype, PageConfig(Layout::TILE), mem_cfg);
    ASSERT_EQ(input_buf_size_datums * datum_size_bytes, tensor_layout.compute_packed_buffer_size_bytes(input_shape));
    auto input_buffer = tt::tt_metal::tensor_impl::allocate_buffer_on_device(device, input_shape, tensor_layout);

    auto input_storage = tt::tt_metal::DeviceStorage{input_buffer};

    Tensor input_tensor = Tensor(input_storage, input_shape, dtype, Layout::TILE);

    ttnn::write_buffer(io_cq, input_tensor, {host_data});

    ttnn::read_buffer(io_cq, input_tensor, {readback_data});

    for (int i = 0; i < input_buf_size_datums; i++) {
        EXPECT_EQ(host_data[i], readback_data[i]);
    }

    input_tensor.deallocate();
}

struct CreateTensorParams {
    ttnn::SimpleShape shape;
};

}

class CreateTensorTest : public ttnn::TTNNFixtureWithDevice, public ::testing::WithParamInterface<CreateTensorParams> {};

TEST_P(CreateTensorTest, Tile) {
    CreateTensorParams params = GetParam();
    run_create_tensor_test(device_, params.shape);
}

INSTANTIATE_TEST_SUITE_P(
    CreateTensorTestWithShape,
    CreateTensorTest,
    ::testing::Values(
        CreateTensorParams{.shape=ttnn::SimpleShape({1, 1, 32, 32})},
        CreateTensorParams{.shape=ttnn::SimpleShape({2, 1, 32, 32})},
        CreateTensorParams{.shape=ttnn::SimpleShape({0, 0, 0, 0})},
        CreateTensorParams{.shape=ttnn::SimpleShape({0, 1, 32, 32})},
        CreateTensorParams{.shape=ttnn::SimpleShape({0})}
    )
);

namespace tt::tt_metal {
inline std::ostream& operator<<(std::ostream& os, Layout layout) {
    switch (layout) {
        case Layout::ROW_MAJOR: os << "ROW_MAJOR"; break;
        case Layout::TILE: os << "TILE"; break;
        case Layout::INVALID: os << "INVALID"; break;
        default: os << "UNKNOWN";
    }
    return os;
}

inline std::ostream& operator<<(std::ostream& os, DataType data_type) {
    switch (data_type) {
        case DataType::BFLOAT16: os << "BFLOAT16"; break;
        case DataType::FLOAT32: os << "FLOAT32"; break;
        case DataType::UINT32: os << "UINT32"; break;
        case DataType::BFLOAT8_B: os << "BFLOAT8_B"; break;
        case DataType::BFLOAT4_B: os << "BFLOAT4_B"; break;
        case DataType::UINT8: os << "UINT8"; break;
        case DataType::UINT16: os << "UINT16"; break;
        case DataType::INT32: os << "INT32"; break;
        case DataType::INVALID: os << "INVALID"; break;
        default: os << "UNKNOWN";
    }
    return os;
}

inline std::ostream& operator<<(std::ostream& os, TensorMemoryLayout layout) {
    switch (layout) {
        case TensorMemoryLayout::INTERLEAVED: os << "INTERLEAVED"; break;
        case TensorMemoryLayout::SINGLE_BANK: os << "SINGLE_BANK"; break;
        case TensorMemoryLayout::HEIGHT_SHARDED: os << "HEIGHT_SHARDED"; break;
        case TensorMemoryLayout::WIDTH_SHARDED: os << "WIDTH_SHARDED"; break;
        case TensorMemoryLayout::BLOCK_SHARDED: os << "BLOCK_SHARDED"; break;
        default: os << "UNKNOWN";
    }
    return os;
}

inline std::ostream& operator<<(std::ostream& os, ShardOrientation orientation) {
    switch (orientation) {
        case ShardOrientation::ROW_MAJOR: os << "ROW_MAJOR"; break;
        case ShardOrientation::COL_MAJOR: os << "COL_MAJOR"; break;
        default: os << "UNKNOWN";
    }
    return os;
}

inline std::ostream& operator<<(std::ostream& os, BufferType buffer) {
    switch (buffer) {
        case BufferType::DRAM: os << "DRAM"; break;
        case BufferType::L1: os << "L1"; break;
        case BufferType::SYSTEM_MEMORY: os << "SYSTEM_MEMORY"; break;
        case BufferType::L1_SMALL: os << "L1_SMALL"; break;
        case BufferType::TRACE: os << "TRACE"; break;
        default: os << "UNKNOWN";
    }
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const ShardSpec& spec) {
    os << "ShardSpec(grid=?"
       << ", shape={" << spec.shape[0] << ", " << spec.shape[1] << "})";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const MemoryConfig& config) {
    os << "MemoryConfig(memory_layout=" << config.memory_layout
       << ", buffer_type=" << config.buffer_type
       << ", shard_spec=";
    if (config.shard_spec) {
        os << *config.shard_spec;
    } else {
        os << "nullopt";
    }
    os << ")";
    return os;
}
}

using CombinationInputParams = std::tuple<ttnn::Shape, tt::tt_metal::DataType, tt::tt_metal::Layout, tt::tt_metal::MemoryConfig>;
class EmptyTensorTest : public ttnn::TTNNFixtureWithDevice, public ::testing::WithParamInterface<CombinationInputParams> {};

TEST_P(EmptyTensorTest, Combinations) {
    auto params = GetParam();
    auto shape = std::get<0>(params);
    auto dtype = std::get<1>(params);
    auto layout = std::get<2>(params);
    auto memory_config = std::get<3>(params);
    tt::log_info("Running test with shape={}, dtype={}, layout={}, memory_config={}", shape, dtype, layout, memory_config);

    if(layout == tt::tt_metal::Layout::ROW_MAJOR && dtype == tt::tt_metal::DataType::BFLOAT8_B) {
        return;
    }

    auto tensor = tt::tt_metal::create_device_tensor(shape, dtype, layout, device_, memory_config);
    EXPECT_EQ(tensor.get_logical_shape(), shape.logical_shape());

    auto tensor_layout = tt::tt_metal::TensorLayout::fromLegacyPaddedShape(dtype, PageConfig(layout), memory_config, shape);
    test_utils::test_tensor_on_device(shape.logical_shape(), tensor_layout, device_);
}

INSTANTIATE_TEST_SUITE_P(
    EmptyTensorTestWithShape,
    EmptyTensorTest,
    ::testing::Combine(
        ::testing::Values(
            //ttnn::Shape({}),
            //ttnn::Shape({0}),
            //ttnn::Shape({1}),
            ttnn::Shape({1, 2}),
            ttnn::Shape({1, 2, 3}),
            ttnn::Shape({1, 2, 3, 4}),
            //ttnn::Shape({0, 0, 0, 0}), fails with width sharded case
            ttnn::Shape({1, 1, 1, 1}),
            //ttnn::Shape({0, 1, 32, 32}), fails with width sharded case
            ttnn::Shape({1, 1, 32, 32}),
            ttnn::Shape({2, 1, 32, 32}),
            ttnn::Shape({64, 1, 256, 1}),
            ttnn::Shape({1, 1, 21120, 16}),
            ttnn::Shape({1, 2, 3, 4, 5})
        ),

        ::testing::Values(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::DataType::FLOAT32,
            tt::tt_metal::DataType::BFLOAT8_B),

        ::testing::Values(
            tt::tt_metal::Layout::TILE,
            tt::tt_metal::Layout::ROW_MAJOR),

        ::testing::Values(
            tt::tt_metal::MemoryConfig{
                .memory_layout = tt::tt_metal::TensorMemoryLayout::SINGLE_BANK,
                .buffer_type = ttnn::BufferType::L1
            },

            tt::tt_metal::MemoryConfig{
                .memory_layout = tt::tt_metal::TensorMemoryLayout::SINGLE_BANK,
                .buffer_type = ttnn::BufferType::DRAM
            },


            tt::tt_metal::MemoryConfig{
                .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
                .buffer_type = tt::tt_metal::BufferType::L1
            },

            tt::tt_metal::MemoryConfig{
                .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
                .buffer_type = tt::tt_metal::BufferType::DRAM
            },


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


            ttnn::MemoryConfig{
                .memory_layout = tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
                .buffer_type = tt::tt_metal::BufferType::L1,
                .shard_spec = ShardSpec{
                    CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{1, 4}}}},
                    {32, 64},
                    tt::tt_metal::ShardOrientation::ROW_MAJOR,
                    false}
            },
            // tt::tt_metal::MemoryConfig{
            //     .memory_layout = tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED,
            //     .buffer_type = tt::tt_metal::BufferType::DRAM,
            //     .shard_spec = ShardSpec{
            //         CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{1, 4}}}},
            //         {32, 64},
            //         tt::tt_metal::ShardOrientation::ROW_MAJOR,
            //         false}
            // },


            tt::tt_metal::MemoryConfig{
                .memory_layout = tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
                .buffer_type = tt::tt_metal::BufferType::L1,
                .shard_spec = ShardSpec{
                    CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{4, 4}}}},
                    {64, 64},
                    tt::tt_metal::ShardOrientation::ROW_MAJOR,
                    false}
            }
            // tt::tt_metal::MemoryConfig{
            //     .memory_layout = tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED,
            //     .buffer_type = tt::tt_metal::BufferType::DRAM,
            //     .shard_spec = ShardSpec{
            //         CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{4, 4}}}},
            //         {64, 64},
            //         tt::tt_metal::ShardOrientation::ROW_MAJOR,
            //         false}
            // }
        )
    )
);
