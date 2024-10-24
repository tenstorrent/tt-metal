// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "common_tensor_test_utils.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/async_runtime.hpp"

#include "gtest/gtest.h"

namespace test_utils {

void test_tensor_on_device(const ttnn::SimpleShape& input_shape, const TensorLayout& layout, tt::tt_metal::Device* device) {
    using namespace tt::tt_metal;

    const uint32_t io_cq = 0;

    const auto input_buf_size_bytes = layout.get_packed_buffer_size_bytes(input_shape);
    const auto host_buffer_datum_size_bytes = sizeof(uint32_t);
    const auto input_buf_size = input_buf_size_bytes / host_buffer_datum_size_bytes;

    auto host_data = std::make_shared<uint32_t[]>(input_buf_size);
    auto readback_data = std::make_shared<uint32_t[]>(input_buf_size);

    const auto random_prime_number = 4051;
    for (int i = 0; i < input_buf_size; i++) {
        host_data[i] = i % random_prime_number;
    }

    auto tensor = tt::tt_metal::create_device_tensor(input_shape, layout, device);
    ttnn::queue_synchronize(device->command_queue(io_cq));

    ttnn::write_buffer(io_cq, tensor, {host_data});
    ttnn::queue_synchronize(device->command_queue(io_cq));

    ttnn::read_buffer(io_cq, tensor, {readback_data});
    ttnn::queue_synchronize(device->command_queue(io_cq));

    for (int i = 0; i < input_buf_size; i++) {
        EXPECT_EQ(host_data[i], readback_data[i]);
    }

    EXPECT_EQ(tensor.get_padded_shape(), layout.get_padded_shape(input_shape));
    tensor.deallocate();
}

void test_tensor_on_device(const ttnn::SimpleShape& input_shape, const tt::tt_metal::TensorLayout& layout) {
    tt::tt_metal::Device* device =  tt::tt_metal::CreateDevice(0);

    test_tensor_on_device(input_shape, layout, device);

    tt::tt_metal::CloseDevice(device);
}

} // namespace test_utils
