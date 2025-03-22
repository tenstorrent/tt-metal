// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn_test_fixtures.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include <tt-metalium/bfloat16.hpp>
#include "ttnn/async_runtime.hpp"
#include "ttnn/operations/functions.hpp"
#include <tt-metalium/event.hpp>
#include <cmath>

using namespace tt;
using namespace tt_metal;
using MultiCommandQueueT3KFixture = ttnn::MultiCommandQueueT3KFixture;

Tensor dispatch_ops_to_device(IDevice* dev, Tensor input_tensor, QueueId cq_id) {
    using ttnn::operations::unary::UnaryOpType;
    using ttnn::operations::unary::UnaryWithParam;

    Tensor output_tensor = ttnn::mul_sfpu(cq_id, input_tensor, 2);
    for (int i = 0; i < 3; i++) {
        output_tensor = ttnn::neg(cq_id, output_tensor);
        output_tensor = ttnn::neg(cq_id, output_tensor);
        output_tensor = ttnn::mul_sfpu(cq_id, output_tensor, 2);
    }
    output_tensor = ttnn::neg(cq_id, output_tensor);
    output_tensor = ttnn::mul_sfpu(cq_id, output_tensor, 2);
    output_tensor = ttnn::add_sfpu(cq_id, output_tensor, 500);
    return output_tensor;
}

TEST_F(MultiCommandQueueT3KFixture, Test2CQMultiDeviceProgramsOnCQ1) {
    // 8 devices with 2 CQs. Enable this test on T3K only.
    if (tt::tt_metal::GetNumAvailableDevices() < 8 or
        tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()) != tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP();
    }

    MemoryConfig mem_cfg = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        .buffer_type = BufferType::DRAM,
        .shard_spec = std::nullopt};

    ttnn::Shape shape{1, 3, 2048, 2048};
    uint32_t buf_size_datums = 2048 * 2048 * 3;
    uint32_t datum_size_bytes = 2;
    auto host_data = std::shared_ptr<bfloat16[]>(new bfloat16[buf_size_datums]);
    auto readback_data = std::shared_ptr<bfloat16[]>(new bfloat16[buf_size_datums]);
    for (int outer_loop = 0; outer_loop < 5; outer_loop++) {
        log_info(LogTest, "Running outer loop {}", outer_loop);
        for (int i = 0; i < 30; i++) {
            for (auto& dev : this->devs) {
                auto dev_idx = dev.first;
                auto device = dev.second;
                if (i == 0 and outer_loop == 0) {
                    device->enable_program_cache();
                }
                for (int j = 0; j < buf_size_datums; j++) {
                    host_data[j] = bfloat16(static_cast<float>(i + dev_idx));
                }
                TensorSpec tensor_spec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), mem_cfg));
                ASSERT_EQ(buf_size_datums * datum_size_bytes, tensor_spec.compute_packed_buffer_size_bytes());
                auto input_tensor = allocate_tensor_on_mesh(tensor_spec, device.get());

                ttnn::write_buffer(
                    ttnn::QueueId(0),
                    input_tensor,
                    {host_data, host_data, host_data, host_data, host_data, host_data, host_data, host_data});
                auto write_event = ttnn::record_event(device->mesh_command_queue(0));
                ttnn::wait_for_event(device->mesh_command_queue(1), write_event);
                auto output_tensor = dispatch_ops_to_device(device.get(), input_tensor, ttnn::QueueId(1));
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

                for (int j = 0; j < 3 * 2048 * 2048; j++) {
                    ASSERT_EQ(readback_data[j].to_float(), -1 * (i + dev_idx) * 32 + 500);
                }
            }
        }
    }
}

TEST_F(MultiCommandQueueT3KFixture, Test2CQMultiDeviceProgramsOnCQ0) {
    // 8 devices with 2 CQs. Enable this test on T3K only.
    if (tt::tt_metal::GetNumAvailableDevices() < 8 or
        tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()) != tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP();
    }

    MemoryConfig mem_cfg = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        .buffer_type = BufferType::DRAM,
        .shard_spec = std::nullopt};

    ttnn::Shape shape{1, 3, 2048, 2048};
    uint32_t buf_size_datums = 2048 * 2048 * 3;
    uint32_t datum_size_bytes = 2;
    auto host_data = std::shared_ptr<bfloat16[]>(new bfloat16[buf_size_datums]);
    auto readback_data = std::shared_ptr<bfloat16[]>(new bfloat16[buf_size_datums]);

    TensorSpec tensor_spec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), mem_cfg));
    ASSERT_EQ(buf_size_datums * datum_size_bytes, tensor_spec.compute_packed_buffer_size_bytes());
    for (int outer_loop = 0; outer_loop < 5; outer_loop++) {
        log_info(LogTest, "Running outer loop {}", outer_loop);
        for (int i = 0; i < 30; i++) {
            for (auto& dev : this->devs) {
                auto dev_idx = dev.first;
                auto device = dev.second;
                if (i == 0 and outer_loop == 0) {
                    device->enable_program_cache();
                }
                for (int j = 0; j < buf_size_datums; j++) {
                    host_data[j] = bfloat16(static_cast<float>(i + dev_idx));
                }
                auto input_tensor = allocate_tensor_on_mesh(tensor_spec, device.get());

                ttnn::write_buffer(
                    ttnn::QueueId(1),
                    input_tensor,
                    {host_data, host_data, host_data, host_data, host_data, host_data, host_data, host_data});
                auto write_event = ttnn::record_event(device->mesh_command_queue(1));
                ttnn::wait_for_event(device->mesh_command_queue(0), write_event);
                auto output_tensor = dispatch_ops_to_device(device.get(), input_tensor, ttnn::DefaultQueueId);
                auto workload_event = ttnn::record_event(device->mesh_command_queue(0));
                ttnn::wait_for_event(device->mesh_command_queue(1), workload_event);
                // std::this_thread::sleep_for(std::chrono::milliseconds(50));
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

                for (int j = 0; j < 3 * 2048 * 2048; j++) {
                    ASSERT_EQ(readback_data[j].to_float(), -1 * (i + dev_idx) * 32 + 500);
                }
            }
        }
    }
}

TEST_F(MultiCommandQueueT3KFixture, Test2CQMultiDeviceWithCQ1Only) {
    // 8 devices with 2 CQs. Enable this test on T3K only.
    if (tt::tt_metal::GetNumAvailableDevices() < 8 or
        tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()) != tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP();
    }

    MemoryConfig mem_cfg = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        .buffer_type = BufferType::DRAM,
        .shard_spec = std::nullopt};

    ttnn::Shape shape{1, 3, 2048, 2048};
    uint32_t buf_size_datums = 2048 * 2048 * 3;
    uint32_t datum_size_bytes = 2;
    auto host_data = std::shared_ptr<bfloat16[]>(new bfloat16[buf_size_datums]);
    auto readback_data = std::shared_ptr<bfloat16[]>(new bfloat16[buf_size_datums]);

    for (int outer_loop = 0; outer_loop < 5; outer_loop++) {
        log_info(LogTest, "Running outer loop {}", outer_loop);
        for (int i = 0; i < 30; i++) {
            for (auto& dev : this->devs) {
                auto dev_idx = dev.first;
                auto device = dev.second;
                if (i == 0 and outer_loop == 0) {
                    device->enable_program_cache();
                }
                for (int j = 0; j < buf_size_datums; j++) {
                    host_data[j] = bfloat16(static_cast<float>(i + dev_idx));
                }

                TensorSpec tensor_spec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), mem_cfg));
                auto input_tensor = allocate_tensor_on_mesh(tensor_spec, device.get());

                ttnn::write_buffer(
                    ttnn::QueueId(1),
                    input_tensor,
                    {host_data, host_data, host_data, host_data, host_data, host_data, host_data, host_data});
                auto write_event = ttnn::record_event(device->mesh_command_queue(1));
                ttnn::wait_for_event(device->mesh_command_queue(1), write_event);
                auto output_tensor = dispatch_ops_to_device(device.get(), input_tensor, ttnn::QueueId(1));
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

                for (int j = 0; j < 3 * 2048 * 2048; j++) {
                    if (readback_data[j].to_float() != -1 * (i + dev_idx) * 32 + 500) {
                        ASSERT_EQ(readback_data[j].to_float(), -1 * (i + dev_idx) * 32 + 500);
                    }
                }
            }
        }
    }
}
