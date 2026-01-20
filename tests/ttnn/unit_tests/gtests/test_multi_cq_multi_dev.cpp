// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include "ttnn/tensor/tensor_ops.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/event.hpp>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "common_test_utils.hpp"
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include "ttnn/async_runtime.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"

#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn_test_fixtures.hpp"
#include <umd/device/types/arch.hpp>

using namespace tt;
using namespace tt_metal;
using MultiCommandQueueT3KFixture = ttnn::MultiCommandQueueT3KFixture;

TEST_F(MultiCommandQueueT3KFixture, Test2CQMultiDeviceProgramsOnCQ1) {
    MemoryConfig mem_cfg = MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};

    ttnn::Shape shape{1, 3, 2048, 2048};
    uint32_t buf_size_datums = 2048 * 2048 * 3;
    uint32_t datum_size_bytes = 2;
    auto host_data = std::shared_ptr<bfloat16[]>(new bfloat16[buf_size_datums]);
    auto readback_data = std::shared_ptr<bfloat16[]>(new bfloat16[buf_size_datums]);
    auto expected_data = std::shared_ptr<bfloat16[]>(new bfloat16[buf_size_datums]);
    for (int outer_loop = 0; outer_loop < 2; outer_loop++) {
        log_info(LogTest, "Running outer loop {}", outer_loop);
        for (int i = 0; i < 5; i++) {
            for (auto& dev : this->devs) {
                auto dev_idx = dev.first;
                auto device = dev.second;

                for (uint32_t j = 0; j < buf_size_datums; j++) {
                    host_data[j] = bfloat16(static_cast<float>(i + dev_idx));
                }
                // Pre-compute expected output: input goes through dispatch_ops_to_device which applies -32x + 128
                float expected_val = (-1 * (i + dev_idx) * 32) + 128;
                for (uint32_t j = 0; j < buf_size_datums; j++) {
                    expected_data[j] = bfloat16(expected_val);
                }

                TensorSpec tensor_spec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), mem_cfg));
                ASSERT_EQ(buf_size_datums * datum_size_bytes, tensor_spec.compute_packed_buffer_size_bytes());
                auto input_tensor = create_device_tensor(tensor_spec, device.get());

                ttnn::write_buffer(
                    ttnn::QueueId(0),
                    input_tensor,
                    {host_data, host_data, host_data, host_data, host_data, host_data, host_data, host_data});
                auto write_event = ttnn::record_event(device->mesh_command_queue(0));
                ttnn::wait_for_event(device->mesh_command_queue(1), write_event);
                auto output_tensor = ttnn::test_utils::dispatch_ops_to_device(input_tensor, ttnn::QueueId(1));
                auto workload_event = ttnn::record_event(device->mesh_command_queue(1));
                ttnn::wait_for_event(device->mesh_command_queue(0), workload_event);

                ttnn::read_buffer(
                    ttnn::QueueId(0),
                    output_tensor,
                    {readback_data,
                     readback_data,
                     readback_data,
                     readback_data,
                     readback_data,
                     readback_data,
                     readback_data,
                     readback_data});

                ASSERT_EQ(std::memcmp(readback_data.get(), expected_data.get(), buf_size_datums * datum_size_bytes), 0)
                    << "Data mismatch at outer_loop " << outer_loop << " iteration " << i << " device " << dev_idx;
            }
        }
    }
}

TEST_F(MultiCommandQueueT3KFixture, Test2CQMultiDeviceProgramsOnCQ0) {
    MemoryConfig mem_cfg = MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};

    ttnn::Shape shape{1, 3, 2048, 2048};
    uint32_t buf_size_datums = 2048 * 2048 * 3;
    uint32_t datum_size_bytes = 2;
    auto host_data = std::shared_ptr<bfloat16[]>(new bfloat16[buf_size_datums]);
    auto readback_data = std::shared_ptr<bfloat16[]>(new bfloat16[buf_size_datums]);
    auto expected_data = std::shared_ptr<bfloat16[]>(new bfloat16[buf_size_datums]);

    TensorSpec tensor_spec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), mem_cfg));
    ASSERT_EQ(buf_size_datums * datum_size_bytes, tensor_spec.compute_packed_buffer_size_bytes());
    for (int outer_loop = 0; outer_loop < 2; outer_loop++) {
        log_info(LogTest, "Running outer loop {}", outer_loop);
        for (int i = 0; i < 5; i++) {
            for (auto& dev : this->devs) {
                auto dev_idx = dev.first;
                auto device = dev.second;

                for (uint32_t j = 0; j < buf_size_datums; j++) {
                    host_data[j] = bfloat16(static_cast<float>(i + dev_idx));
                }

                // Pre-compute expected output: input goes through dispatch_ops_to_device which applies -32x + 128
                float expected_val = (-1 * (i + dev_idx) * 32) + 128;
                for (uint32_t j = 0; j < buf_size_datums; j++) {
                    expected_data[j] = bfloat16(expected_val);
                }

                auto input_tensor = create_device_tensor(tensor_spec, device.get());

                ttnn::write_buffer(
                    ttnn::QueueId(1),
                    input_tensor,
                    {host_data, host_data, host_data, host_data, host_data, host_data, host_data, host_data});
                auto write_event = ttnn::record_event(device->mesh_command_queue(1));
                ttnn::wait_for_event(device->mesh_command_queue(0), write_event);
                auto output_tensor = ttnn::test_utils::dispatch_ops_to_device(input_tensor, QueueId(0));
                auto workload_event = ttnn::record_event(device->mesh_command_queue(0));
                ttnn::wait_for_event(device->mesh_command_queue(1), workload_event);
                ttnn::read_buffer(
                    ttnn::QueueId(1),
                    output_tensor,
                    {readback_data,
                     readback_data,
                     readback_data,
                     readback_data,
                     readback_data,
                     readback_data,
                     readback_data,
                     readback_data});

                ASSERT_EQ(std::memcmp(readback_data.get(), expected_data.get(), buf_size_datums * datum_size_bytes), 0)
                    << "Data mismatch at outer_loop " << outer_loop << " iteration " << i << " device " << dev_idx;
            }
        }
    }
}

TEST_F(MultiCommandQueueT3KFixture, Test2CQMultiDeviceWithCQ1Only) {
    MemoryConfig mem_cfg = MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};

    ttnn::Shape shape{1, 3, 2048, 2048};
    uint32_t buf_size_datums = 2048 * 2048 * 3;
    uint32_t datum_size_bytes = 2;
    auto host_data = std::shared_ptr<bfloat16[]>(new bfloat16[buf_size_datums]);
    auto readback_data = std::shared_ptr<bfloat16[]>(new bfloat16[buf_size_datums]);
    auto expected_data = std::shared_ptr<bfloat16[]>(new bfloat16[buf_size_datums]);

    for (int outer_loop = 0; outer_loop < 2; outer_loop++) {
        log_info(LogTest, "Running outer loop {}", outer_loop);
        for (int i = 0; i < 5; i++) {
            for (auto& dev : this->devs) {
                auto dev_idx = dev.first;
                auto device = dev.second;
                for (uint32_t j = 0; j < buf_size_datums; j++) {
                    host_data[j] = bfloat16(static_cast<float>(i + dev_idx));
                }
                // Pre-compute expected output: input goes through dispatch_ops_to_device which applies -32x + 128
                float expected_val = (-1 * (i + dev_idx) * 32) + 128;
                for (uint32_t j = 0; j < buf_size_datums; j++) {
                    expected_data[j] = bfloat16(expected_val);
                }

                TensorSpec tensor_spec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), mem_cfg));
                auto input_tensor = create_device_tensor(tensor_spec, device.get());

                ttnn::write_buffer(
                    ttnn::QueueId(1),
                    input_tensor,
                    {host_data, host_data, host_data, host_data, host_data, host_data, host_data, host_data});
                auto write_event = ttnn::record_event(device->mesh_command_queue(1));
                ttnn::wait_for_event(device->mesh_command_queue(1), write_event);
                auto output_tensor = ttnn::test_utils::dispatch_ops_to_device(input_tensor, ttnn::QueueId(1));
                auto workload_event = ttnn::record_event(device->mesh_command_queue(1));
                ttnn::wait_for_event(device->mesh_command_queue(1), workload_event);
                ttnn::read_buffer(
                    ttnn::QueueId(1),
                    output_tensor,
                    {readback_data,
                     readback_data,
                     readback_data,
                     readback_data,
                     readback_data,
                     readback_data,
                     readback_data,
                     readback_data});

                ASSERT_EQ(std::memcmp(readback_data.get(), expected_data.get(), buf_size_datums * datum_size_bytes), 0)
                    << "Data mismatch at outer_loop " << outer_loop << " iteration " << i << " device " << dev_idx;
            }
        }
    }
}
