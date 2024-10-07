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

#include "ttnn_test_fixtures.hpp"

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

    auto input_buffer = ttnn::allocate_buffer_on_device(input_buf_size_datums * datum_size_bytes, device, input_shape, dtype, Layout::TILE, mem_cfg);

    auto input_storage = tt::tt_metal::DeviceStorage{input_buffer};

    Tensor input_tensor = Tensor(input_storage, input_shape, dtype, Layout::TILE);
    tt::log_debug("input_data: \n {}", input_tensor.write_to_string());

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

class CreateTensorTest : public ttnn::TTNNFixtureWithDevice, public ::testing::WithParamInterface<CreateTensorParams> {};

TEST_P(CreateTensorTest, Tile) {
    CreateTensorParams params = GetParam();
    run_create_tensor_test(device_, params.shape);
}

INSTANTIATE_TEST_SUITE_P(
    CreateTensorTestWithShape,
    CreateTensorTest,
    ::testing::Values(
        CreateTensorParams{.shape=ttnn::SimpleShape({1, 1, 32, 32})}
    )
);
