// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tests for TransferToDevice/TransferToHost API - round-trip data transfer between HostTensor and DeviceTensor.

#include <gtest/gtest.h>
#include <numeric>
#include <vector>

#include <tt-metalium/mesh_command_queue.hpp>

#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/experimental/tensor/device_tensor.hpp>
#include <tt-metalium/experimental/tensor/tensor_apis.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::test {
namespace {

using namespace tt::tt_metal;

// Single device (1x1 mesh) fixture for testing tensor APIs
class SingleDeviceMeshFixture : public MeshDeviceFixtureBase {
protected:
    SingleDeviceMeshFixture() :
        MeshDeviceFixtureBase(Config{.mesh_shape = distributed::MeshShape{1, 1}, .num_cqs = 1}) {}
};

using TransferToDeviceTest = SingleDeviceMeshFixture;

// TODO: This should be a public utlity:
// Helper function to create a HostTensor for receiving data from device.
// Creates a HostTensor with pre-allocated float buffer using from_vector.
HostTensor create_receiving_host_tensor(const TensorSpec& tensor_spec) {
    const size_t num_elements = tensor_spec.logical_shape().volume();
    std::vector<float> data(num_elements, 0.0f);
    return HostTensor::from_vector(std::move(data), tensor_spec);
}

// Helper function to extract float data from a HostTensor's buffer directly.
// This works for ROW_MAJOR layout where logical shape matches physical shape.
std::vector<float> get_float_data_from_host_tensor(const HostTensor& tensor) {
    auto buffer = tensor.get_host_buffer();
    auto data_span = buffer.view_as<const float>();
    return std::vector<float>(data_span.begin(), data_span.end());
}

TEST_F(TransferToDeviceTest, BasicRoundTrip) {
    // Create host tensor with some data
    const Shape shape{1, 1, 32, 32};
    const size_t num_elements = shape.volume();

    std::vector<float> host_data(num_elements);
    std::iota(host_data.begin(), host_data.end(), 0.0f);

    TensorSpec tensor_spec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{}));

    // Create host tensor using from_vector
    auto host_tensor = HostTensor::from_vector(std::vector<float>(host_data), tensor_spec);

    // Create device tensor spec with DRAM memory config and allocate on device
    TensorSpec device_tensor_spec(
        shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{BufferType::DRAM}));
    auto device_tensor = DeviceTensor::allocate_on_device(device_tensor_spec, *mesh_device_);

    ASSERT_TRUE(device_tensor.is_allocated());

    // Transfer data to device
    auto& cq = mesh_device_->mesh_command_queue();
    TransferToDevice(cq, host_tensor, device_tensor, /*blocking=*/true);

    // Create receiving host tensor and transfer back from device
    auto result_tensor = create_receiving_host_tensor(device_tensor_spec);
    TransferToHost(cq, device_tensor, result_tensor, /*blocking=*/true);

    // Compare round-tripped data with original
    auto result_data = get_float_data_from_host_tensor(result_tensor);
    EXPECT_EQ(result_data, host_data);
}

// Test that topology is preserved through round-trip transfer
TEST_F(TransferToDeviceTest, TopologyPreservedAfterRoundTrip) {
    const Shape shape{1, 1, 32, 32};
    const size_t num_elements = shape.volume();

    std::vector<float> host_data(num_elements);
    std::iota(host_data.begin(), host_data.end(), 0.0f);

    TensorSpec tensor_spec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{}));

    auto host_tensor = HostTensor::from_vector(std::vector<float>(host_data), tensor_spec);

    TensorSpec device_tensor_spec(
        shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{BufferType::DRAM}));
    auto device_tensor = DeviceTensor::allocate_on_device(device_tensor_spec, *mesh_device_);

    auto& cq = mesh_device_->mesh_command_queue();
    TransferToDevice(cq, host_tensor, device_tensor, /*blocking=*/true);

    // Verify DeviceTensor is still valid after transfer (storage/topology were updated)
    EXPECT_TRUE(device_tensor.is_allocated());
    EXPECT_EQ(device_tensor.logical_shape(), shape);
    EXPECT_EQ(device_tensor.dtype(), DataType::FLOAT32);

    // Transfer back to host and verify data integrity
    auto result_tensor = create_receiving_host_tensor(device_tensor_spec);
    TransferToHost(cq, device_tensor, result_tensor, /*blocking=*/true);

    // Verify the received HostTensor has correct properties
    EXPECT_EQ(result_tensor.logical_shape(), shape);
    EXPECT_EQ(result_tensor.dtype(), DataType::FLOAT32);

    // Verify data is correct
    auto result_data = get_float_data_from_host_tensor(result_tensor);
    EXPECT_EQ(result_data, host_data);
}

// Test round-trip with different data values to ensure data integrity
TEST_F(TransferToDeviceTest, DataIntegrityRoundTrip) {
    const Shape shape{1, 1, 32, 32};
    const size_t num_elements = shape.volume();

    // Use non-sequential data to better test data integrity
    std::vector<float> host_data(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
        host_data[i] = static_cast<float>(i * 3.14159f + 1.0f);
    }

    TensorSpec tensor_spec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{}));

    auto host_tensor = HostTensor::from_vector(std::vector<float>(host_data), tensor_spec);

    TensorSpec device_tensor_spec(
        shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{BufferType::DRAM}));
    auto device_tensor = DeviceTensor::allocate_on_device(device_tensor_spec, *mesh_device_);

    ASSERT_TRUE(device_tensor.is_allocated());

    auto& cq = mesh_device_->mesh_command_queue();
    TransferToDevice(cq, host_tensor, device_tensor, /*blocking=*/true);

    auto result_tensor = create_receiving_host_tensor(device_tensor_spec);
    TransferToHost(cq, device_tensor, result_tensor, /*blocking=*/true);

    auto result_data = get_float_data_from_host_tensor(result_tensor);
    EXPECT_EQ(result_data, host_data);
}

// Test round-trip with L1 buffer type
TEST_F(TransferToDeviceTest, L1BufferRoundTrip) {
    const Shape shape{1, 1, 32, 32};
    const size_t num_elements = shape.volume();

    std::vector<float> host_data(num_elements);
    std::iota(host_data.begin(), host_data.end(), 0.0f);

    TensorSpec tensor_spec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{}));

    auto host_tensor = HostTensor::from_vector(std::vector<float>(host_data), tensor_spec);

    // Use L1 buffer type instead of DRAM
    TensorSpec device_tensor_spec(
        shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{BufferType::L1}));
    auto device_tensor = DeviceTensor::allocate_on_device(device_tensor_spec, *mesh_device_);

    ASSERT_TRUE(device_tensor.is_allocated());

    auto& cq = mesh_device_->mesh_command_queue();
    TransferToDevice(cq, host_tensor, device_tensor, /*blocking=*/true);

    auto result_tensor = create_receiving_host_tensor(device_tensor_spec);
    TransferToHost(cq, device_tensor, result_tensor, /*blocking=*/true);

    auto result_data = get_float_data_from_host_tensor(result_tensor);
    EXPECT_EQ(result_data, host_data);
}

}  // namespace
}  // namespace tt::tt_metal::test
