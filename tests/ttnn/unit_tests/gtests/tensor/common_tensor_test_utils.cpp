// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "common_tensor_test_utils.hpp"

#include <stdint.h>
#include <memory>
#include <vector>

#include <tt-metalium/device.hpp>
#include "gtest/gtest.h"
#include <tt-metalium/host_api.hpp>
#include "ttnn/async_runtime.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

namespace tt {
namespace tt_metal {
class Shape;
}  // namespace tt_metal
}  // namespace tt

namespace test_utils {

void test_tensor_on_device(
    const ttnn::Shape& input_shape,
    const tt::tt_metal::TensorLayout& layout,
    tt::tt_metal::distributed::MeshDevice* device) {
    using namespace tt::tt_metal;

    const ttnn::QueueId io_cq = ttnn::DefaultQueueId;

    const auto input_buf_size_bytes = layout.compute_packed_buffer_size_bytes(input_shape);
    const auto host_buffer_datum_size_bytes = sizeof(uint32_t);
    const auto input_buf_size = input_buf_size_bytes / host_buffer_datum_size_bytes;

    auto host_data = std::shared_ptr<void>(new uint32_t[input_buf_size], std::default_delete<uint32_t[]>());
    auto* host_data_ptr = static_cast<uint32_t*>(host_data.get());

    auto readback_data = std::shared_ptr<void>(new uint32_t[input_buf_size], std::default_delete<uint32_t[]>());
    auto* readback_data_ptr = static_cast<uint32_t*>(readback_data.get());

    const auto random_prime_number = 4051;
    for (int i = 0; i < input_buf_size; i++) {
        host_data_ptr[i] = i % random_prime_number;
    }

    auto tensor = tt::tt_metal::create_device_tensor(TensorSpec(input_shape, layout), device);
    ttnn::queue_synchronize(device->mesh_command_queue(*io_cq));

    ttnn::write_buffer(io_cq, tensor, {host_data});
    ttnn::queue_synchronize(device->mesh_command_queue(*io_cq));

    ttnn::read_buffer(io_cq, tensor, {readback_data});
    ttnn::queue_synchronize(device->mesh_command_queue(*io_cq));

    for (int i = 0; i < input_buf_size; i++) {
        EXPECT_EQ(host_data_ptr[i], readback_data_ptr[i]);
        if (host_data_ptr[i] != readback_data_ptr[i]) {
            break;
        }
    }

    EXPECT_EQ(tensor.padded_shape(), layout.compute_padded_shape(input_shape));
    tensor.deallocate();
}

void test_tensor_on_device(const ttnn::Shape& input_shape, const tt::tt_metal::TensorLayout& layout) {
    auto device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
    test_tensor_on_device(input_shape, layout, device.get());
}

}  // namespace test_utils
