// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tests for TransferToDevice API - transferring data from HostTensor to DeviceTensor.

#include <gtest/gtest.h>
#include <numeric>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/host_buffer.hpp>

#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <tt-metalium/experimental/tensor/device_tensor.hpp>
#include <tt-metalium/experimental/tensor/tensor_apis.hpp>
#include <tt-metalium/experimental/tensor/spec/tensor_spec.hpp>
#include <tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::test {
namespace {

using namespace tt::tt_metal;

// Use GenericMeshDeviceFixture for proper multi-device handling (same as test_tensor_topology.cpp)
using TransferToDeviceTest = GenericMeshDeviceFixture;

TEST_F(TransferToDeviceTest, BasicTransfer) {
    // Create host tensor with some data
    const Shape shape{1, 1, 32, 32};
    const size_t num_elements = shape.volume();

    std::vector<float> host_data(num_elements);
    std::iota(host_data.begin(), host_data.end(), 0.0f);

    TensorSpec tensor_spec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{}));

    // Create host tensor using HostBuffer directly
    std::vector<float> data_copy(host_data);
    HostBuffer host_buffer{std::move(data_copy)};
    HostTensor host_tensor{std::move(host_buffer), tensor_spec, TensorTopology{}};

    // Create device tensor spec with DRAM memory config
    TensorSpec device_tensor_spec(
        shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{BufferType::DRAM}));

    // Allocate mesh buffer for device tensor
    distributed::ReplicatedBufferConfig global_config{.size = device_tensor_spec.compute_packed_buffer_size_bytes()};
    distributed::DeviceLocalBufferConfig local_config{
        .page_size = device_tensor_spec.compute_packed_buffer_size_bytes(), .buffer_type = BufferType::DRAM};
    auto mesh_buffer = distributed::MeshBuffer::create(global_config, local_config, mesh_device_.get());

    // Create DeviceTensor from the allocated buffer
    DeviceStorage device_storage(mesh_buffer, {distributed::MeshCoordinate{0, 0}});
    DeviceTensor device_tensor(std::move(device_storage), device_tensor_spec, TensorTopology{});

    ASSERT_TRUE(device_tensor.is_allocated());

    // Transfer data to device
    auto& cq = mesh_device_->mesh_command_queue();
    TransferToDevice(cq, host_tensor, device_tensor, /*blocking=*/true);

    // Read data back and verify
    std::vector<float> result_data(num_elements);
    distributed::ReadShard(cq, result_data, device_tensor.mesh_buffer(), distributed::MeshCoordinate{0, 0});

    EXPECT_EQ(result_data, host_data);
}

// Test that DeviceTensor's topology is updated after transfer
TEST_F(TransferToDeviceTest, TopologyUpdatedAfterTransfer) {
    const Shape shape{1, 1, 32, 32};
    const size_t num_elements = shape.volume();

    std::vector<float> host_data(num_elements);
    std::iota(host_data.begin(), host_data.end(), 0.0f);

    TensorSpec tensor_spec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{}));

    std::vector<float> data_copy(host_data);
    HostBuffer host_buffer{std::move(data_copy)};
    HostTensor host_tensor{std::move(host_buffer), tensor_spec, TensorTopology{}};

    TensorSpec device_tensor_spec(
        shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{BufferType::DRAM}));

    distributed::ReplicatedBufferConfig global_config{.size = device_tensor_spec.compute_packed_buffer_size_bytes()};
    distributed::DeviceLocalBufferConfig local_config{
        .page_size = device_tensor_spec.compute_packed_buffer_size_bytes(), .buffer_type = BufferType::DRAM};
    auto mesh_buffer = distributed::MeshBuffer::create(global_config, local_config, mesh_device_.get());

    DeviceStorage device_storage(mesh_buffer, {distributed::MeshCoordinate{0, 0}});
    DeviceTensor device_tensor(std::move(device_storage), device_tensor_spec, TensorTopology{});

    auto& cq = mesh_device_->mesh_command_queue();
    TransferToDevice(cq, host_tensor, device_tensor, /*blocking=*/true);

    // Verify DeviceTensor is still valid after transfer (storage/topology were updated)
    EXPECT_TRUE(device_tensor.is_allocated());
    EXPECT_EQ(device_tensor.logical_shape(), shape);
    EXPECT_EQ(device_tensor.dtype(), DataType::FLOAT32);
}

// Test replication: 1x1 host tensor replicated to full mesh device
// This test validates replication behavior when the mesh has more than one device.
TEST_F(TransferToDeviceTest, ReplicationToMultiDeviceMesh) {
    const auto mesh_shape = mesh_device_->shape();
    if (mesh_shape.mesh_size() < 2) {
        GTEST_SKIP() << "Need at least 2 devices for replication test";
    }

    const Shape shape{1, 1, 32, 32};
    const size_t num_elements = shape.volume();

    std::vector<float> host_data(num_elements);
    std::iota(host_data.begin(), host_data.end(), 0.0f);

    TensorSpec tensor_spec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{}));

    // Create host tensor on 1x1 mesh (single shard)
    std::vector<float> data_copy(host_data);
    HostBuffer host_buffer{std::move(data_copy)};
    HostTensor host_tensor{std::move(host_buffer), tensor_spec, TensorTopology{}};

    // Create device tensor on the full mesh
    TensorSpec device_tensor_spec(
        shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), MemoryConfig{BufferType::DRAM}));

    distributed::ReplicatedBufferConfig global_config{.size = device_tensor_spec.compute_packed_buffer_size_bytes()};
    distributed::DeviceLocalBufferConfig local_config{
        .page_size = device_tensor_spec.compute_packed_buffer_size_bytes(), .buffer_type = BufferType::DRAM};
    auto mesh_buffer = distributed::MeshBuffer::create(global_config, local_config, mesh_device_.get());

    // Build coords for all devices in the mesh
    std::vector<distributed::MeshCoordinate> all_coords;
    for (size_t row = 0; row < mesh_shape[0]; ++row) {
        for (size_t col = 0; col < mesh_shape[1]; ++col) {
            all_coords.emplace_back(row, col);
        }
    }

    DeviceStorage device_storage(mesh_buffer, all_coords);
    DeviceTensor device_tensor(std::move(device_storage), device_tensor_spec, TensorTopology{});

    ASSERT_TRUE(device_tensor.is_allocated());

    // Transfer data - this should replicate to all devices
    auto& cq = mesh_device_->mesh_command_queue();
    TransferToDevice(cq, host_tensor, device_tensor, /*blocking=*/true);

    // Read data back from all devices and verify replication
    for (size_t i = 0; i < all_coords.size(); ++i) {
        std::vector<float> result_data(num_elements);
        distributed::ReadShard(cq, result_data, device_tensor.mesh_buffer(), all_coords[i]);
        EXPECT_EQ(result_data, host_data) << "Data mismatch at coord index " << i;
    }
}

}  // namespace
}  // namespace tt::tt_metal::test
