// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "ttnn/tensor/tensor_ops.hpp"
#include <cstdint>
#include <tt-metalium/event.hpp>
#include <algorithm>
#include <memory>
#include <numeric>
#include <optional>
#include <thread>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include "gmock/gmock.h"
#include <tt-metalium/shape.hpp>
#include "ttnn/async_runtime.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace tt::tt_metal {
namespace {

using ::tt::tt_metal::is_device_tensor;

using MultiProducerCommandQueueTest = ttnn::MultiCommandQueueSingleDeviceFixture;

TEST_F(MultiProducerCommandQueueTest, Stress) {
    // Spawn 2 application level threads intefacing with the same device through the async engine.
    // This leads to shared access of the work_executor and host side worker queue.
    // Test thread safety.
    auto* device = this->device_;

    const ttnn::Shape tensor_shape{1, 1, 256, 256};
    const MemoryConfig mem_cfg = MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    const TensorLayout tensor_layout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), mem_cfg);
    const TensorSpec tensor_spec(tensor_shape, tensor_layout);

    // Thread 0 uses cq_0, thread 1 uses cq_1
    const ttnn::QueueId t0_io_cq = ttnn::QueueId(0);
    const ttnn::QueueId t1_io_cq = ttnn::QueueId(1);

    std::vector<float> t0_host_data(tensor_shape.volume());
    std::vector<float> t1_host_data(tensor_shape.volume());
    std::iota(t0_host_data.begin(), t0_host_data.end(), 1024);
    std::iota(t1_host_data.begin(), t1_host_data.end(), 2048);

    const Tensor t0_host_tensor = Tensor::from_vector(t0_host_data, tensor_spec);
    const Tensor t1_host_tensor = Tensor::from_vector(t1_host_data, tensor_spec);

    constexpr int kNumIterations = 25;
    std::thread t0([&]() {
        for (int j = 0; j < kNumIterations; j++) {
            Tensor t0_tensor = t0_host_tensor.to_device(device, mem_cfg, t0_io_cq);
            EXPECT_TRUE(is_device_tensor(t0_tensor));
            EXPECT_EQ(t0_tensor.to_vector<float>(t0_io_cq), t0_host_data);
        }
    });

    std::thread t1([&]() {
        for (int j = 0; j < kNumIterations; j++) {
            Tensor t1_tensor = t1_host_tensor.to_device(device, mem_cfg, t1_io_cq);
            EXPECT_TRUE(is_device_tensor(t1_tensor));
            EXPECT_EQ(t1_tensor.to_vector<float>(t1_io_cq), t1_host_data);
        }
    });

    t0.join();
    t1.join();
}

TEST_F(MultiProducerCommandQueueTest, EventSync) {
    // Verify that the event_synchronize API stalls the calling thread until
    // the device records the event being polled.
    // Thread 0 = writer thread. Thread 1 = reader thread.
    // Reader cannot read until writer has correctly updated a memory location.
    // Writer cannot update location until reader has picked up data.
    // Use write_event to stall reader and read_event to stall writer.
    auto* device = this->device_;

    const ttnn::Shape tensor_shape{1, 1, 1024, 1024};
    const MemoryConfig mem_cfg = MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    const TensorLayout tensor_layout(DataType::UINT32, PageConfig(Layout::ROW_MAJOR), mem_cfg);
    const TensorSpec tensor_spec(tensor_shape, tensor_layout);

    const ttnn::QueueId write_cq = ttnn::QueueId(0);
    const ttnn::QueueId read_cq = ttnn::QueueId(1);

    std::optional<tt::tt_metal::distributed::MeshEvent> write_event;
    std::optional<tt::tt_metal::distributed::MeshEvent> read_event;
    std::mutex event_mutex;

    Tensor device_tensor = create_device_tensor(tensor_spec, device);

    constexpr int kNumIterations = 100;
    std::thread t0([&]() {
        std::vector<uint32_t> host_data(tensor_shape.volume());
        for (int j = 0; j < kNumIterations; j++) {
            if (j != 0) {
                while (true) {
                    std::unique_lock<std::mutex> lock(event_mutex);
                    if (read_event.has_value()) {
                        break;
                    }
                    lock.unlock();
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                }
                ttnn::event_synchronize(*read_event);
            }
            {
                std::unique_lock<std::mutex> lock(event_mutex);
                read_event = std::nullopt;
            }

            // Create tensor and transfer to device
            std::iota(host_data.begin(), host_data.end(), j);
            const Tensor host_tensor = Tensor::from_vector(host_data, tensor_spec);
            memcpy(device->mesh_command_queue(*write_cq), device_tensor, host_tensor);
            EXPECT_TRUE(is_device_tensor(device_tensor));

            {
                std::unique_lock<std::mutex> lock(event_mutex);
                write_event = ttnn::record_event_to_host(device->mesh_command_queue(*write_cq));
            }
        }
    });

    std::thread t1([&]() {
        std::vector<uint32_t> expected_readback_data(tensor_shape.volume());
        for (int j = 0; j < kNumIterations; j++) {
            while (true) {
                std::unique_lock<std::mutex> lock(event_mutex);
                if (write_event.has_value()) {
                    break;
                }
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
            ttnn::event_synchronize(*write_event);
            {
                std::unique_lock<std::mutex> lock(event_mutex);
                write_event = std::nullopt;
            }

            // Read back from device and verify
            const Tensor readback_tensor = device_tensor.cpu(/*blocking=*/true, read_cq);
            EXPECT_FALSE(is_device_tensor(readback_tensor));
            std::iota(expected_readback_data.begin(), expected_readback_data.end(), j);
            EXPECT_EQ(readback_tensor.to_vector<uint32_t>(), expected_readback_data) << "At iteration " << j;

            {
                std::unique_lock<std::mutex> lock(event_mutex);
                read_event = ttnn::record_event_to_host(device->mesh_command_queue(*read_cq));
            }
        }
    });

    t0.join();
    t1.join();
}

}  // namespace
}  // namespace tt::tt_metal
