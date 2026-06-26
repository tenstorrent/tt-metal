// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common_tensor_test_utils.hpp"

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>
#include <tt-metalium/experimental/tensor/tensor_apis.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/topology/tensor_topology.hpp>

namespace test_utils {

void test_tensor_on_device(
    const tt::tt_metal::Shape& input_shape,
    const tt::tt_metal::TensorLayout& layout,
    tt::tt_metal::distributed::MeshDevice& device) {
    using namespace tt::tt_metal;

    const auto input_buf_size = layout.compute_packed_buffer_size_bytes(input_shape);

    std::vector<std::byte> host_data(input_buf_size);
    std::vector<std::byte> readback_data(input_buf_size);

    constexpr int random_prime_number = 4051;
    for (size_t i = 0; i < input_buf_size; i++) {
        host_data[i] = static_cast<std::byte>(i % random_prime_number);
    }

    auto tensor = MeshTensor::allocate_on_device(device, TensorSpec(input_shape, layout), TensorTopology());

    auto& cq = device.mesh_command_queue();
    enqueue_write_tensor(cq, host_data.data(), tensor);
    enqueue_read_tensor(cq, tensor, readback_data.data());

    for (size_t i = 0; i < input_buf_size; i++) {
        EXPECT_EQ(host_data[i], readback_data[i]);
        if (host_data[i] != readback_data[i]) {
            break;
        }
    }

    EXPECT_EQ(tensor.padded_shape(), layout.compute_padded_shape(input_shape));
}

void test_tensor_on_device(const tt::tt_metal::Shape& input_shape, const tt::tt_metal::TensorLayout& layout) {
    auto device = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
    test_tensor_on_device(input_shape, layout, *device);
}

}  // namespace test_utils
