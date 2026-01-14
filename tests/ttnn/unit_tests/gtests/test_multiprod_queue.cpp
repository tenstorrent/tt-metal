// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
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

}  // namespace
}  // namespace tt::tt_metal
