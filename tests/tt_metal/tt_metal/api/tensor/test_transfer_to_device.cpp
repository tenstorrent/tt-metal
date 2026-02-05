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

namespace tt::tt_metal::test {
namespace {

using namespace tt::tt_metal;

// Can I use an existing test driver?
class TransferToDeviceTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (GetNumAvailableDevices() == 0) {
            GTEST_SKIP() << "No devices available";
        }

        // Create a single device mesh
        std::vector<tt::ChipId> ids = {0};
        auto devices = distributed::MeshDevice::create_unit_meshes(ids);
        mesh_device_ = std::move(devices[0]);
    }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
        }
    }

    std::shared_ptr<distributed::MeshDevice> mesh_device_;
};

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
    distributed::ReadShard(cq, result_data, mesh_buffer, distributed::MeshCoordinate{0, 0});

    EXPECT_EQ(result_data, host_data);
}

}  // namespace
}  // namespace tt::tt_metal::test
