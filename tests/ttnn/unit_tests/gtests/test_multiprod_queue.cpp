// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/tensor.hpp"
#include "ttnn_multi_command_queue_fixture.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_sum/moreh_sum_op.hpp"
#include "common/bfloat16.hpp"
#include "ttnn/async_runtime.hpp"
#include "tt_numpy/functions.hpp"
#include "tt_metal/impl/event/event.hpp"
#include <cmath>
#include <thread>

using namespace tt;
using namespace tt_metal;
using MultiCommandQueueSingleDeviceFixture = ttnn::MultiCommandQueueSingleDeviceFixture;
using namespace constants;

TEST_F(MultiCommandQueueSingleDeviceFixture, TestMultiProducerLockBasedQueue) {
    // Spawn 2 application level threads intefacing with the same device through the async engine.
    // This leads to shared access of the work_executor and host side worker queue.
    // Test thread safety.
    Device* device = this->device_;
    // Enable async engine and set queue setting to lock_based
    device->set_worker_mode(WorkExecutorMode::ASYNCHRONOUS);
    device->set_worker_queue_mode(WorkerQueueMode::LOCKBASED);

    MemoryConfig mem_cfg = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        .buffer_type = BufferType::DRAM,
        .shard_spec = std::nullopt
    };
    // Thread 0 uses cq_0, thread 1 uses cq_1
    uint32_t t0_io_cq = 0;
    uint32_t t1_io_cq = 1;
    uint32_t tensor_buf_size = 1024 * 1024;
    uint32_t datum_size_bytes = 2;

    ttnn::Shape tensor_shape = ttnn::Shape(tt::tt_metal::LegacyShape({1, 1, 1024, 1024}));
    auto t0_host_data = std::shared_ptr<bfloat16 []>(new bfloat16[tensor_buf_size]);
    auto t0_readback_data = std::shared_ptr<bfloat16 []>(new bfloat16[tensor_buf_size]);
    auto t1_host_data = std::shared_ptr<bfloat16 []>(new bfloat16[tensor_buf_size]);
    auto t1_readback_data = std::shared_ptr<bfloat16 []>(new bfloat16[tensor_buf_size]);

    // Application level threads issue writes and readbacks.
    std::thread t0([&]() {
        for (int j = 0; j < 100; j++) {
            // Initialize data
            for (int i = 0; i < tensor_buf_size; i++) {
                t0_host_data[i] = bfloat16(static_cast<float>(2 + j));
            }
            // Allocate and write buffer
            auto t0_input_buffer = ttnn::allocate_buffer_on_device(tensor_buf_size * datum_size_bytes, device, tensor_shape, DataType::BFLOAT16, Layout::TILE, mem_cfg);
            auto t0_input_storage = tt::tt_metal::DeviceStorage{t0_input_buffer};
            Tensor t0_input_tensor = Tensor(t0_input_storage, tensor_shape, DataType::BFLOAT16, Layout::TILE);
            ttnn::write_buffer(t0_io_cq, t0_input_tensor, {t0_host_data});
            // Readback and verify
            ttnn::read_buffer(t0_io_cq, t0_input_tensor, {t0_readback_data});
            t0_input_tensor.deallocate();
            for (int i = 0; i < tensor_buf_size; i++) {
                EXPECT_EQ(t0_readback_data[i], t0_host_data[i]);

            }
        }
    });

    std::thread t1([&]() {
        for (int j = 0; j < 100; j++) {
            for (int i = 0; i < tensor_buf_size; i++) {
                t1_host_data[i] = bfloat16(static_cast<float>(4 + j));
            }
            auto t1_input_buffer = ttnn::allocate_buffer_on_device(tensor_buf_size * datum_size_bytes, device, tensor_shape, DataType::BFLOAT16, Layout::TILE, mem_cfg);
            auto t1_input_storage = tt::tt_metal::DeviceStorage{t1_input_buffer};
            Tensor t1_input_tensor = Tensor(t1_input_storage, tensor_shape, DataType::BFLOAT16, Layout::TILE);


            ttnn::write_buffer(t1_io_cq, t1_input_tensor, {t1_host_data});
            ttnn::read_buffer(t1_io_cq, t1_input_tensor, {t1_readback_data});

            t1_input_tensor.deallocate();
            for (int i = 0; i < tensor_buf_size; i++) {
                EXPECT_EQ(t1_readback_data[i], t1_host_data[i]);
            }
        }
    });

    t0.join();
    t1.join();

}

TEST_F(MultiCommandQueueSingleDeviceFixture, TestMultiAppThreadSync) {
    // Verify that the event_synchronize API stalls the calling thread until
    // the device records the event being polled.
    // Thread 0 = writer thread. Thread 1 = reader thread.
    // Reader cannot read until writer has correctly updated a memory location.
    // Writer cannot update location until reader has picked up data.
    // Use write_event to stall reader and read_event to stall writer.
    Device* device = this->device_;
    // Enable async engine and set queue setting to lock_based
    device->set_worker_mode(WorkExecutorMode::ASYNCHRONOUS);
    device->set_worker_queue_mode(WorkerQueueMode::LOCKBASED);

    MemoryConfig mem_cfg = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        .buffer_type = BufferType::DRAM,
        .shard_spec = std::nullopt
    };
    uint32_t write_cq = 0;
    uint32_t read_cq = 0;
    uint32_t tensor_buf_size = 1024 * 1024;
    uint32_t datum_size_bytes = 2;

    std::shared_ptr<Event> write_event = std::make_shared<Event>();
    std::shared_ptr<Event> read_event = std::make_shared<Event>();

    ttnn::Shape tensor_shape = ttnn::Shape(tt::tt_metal::LegacyShape({1, 1, 1024, 1024}));
    auto host_data = std::shared_ptr<bfloat16 []>(new bfloat16[tensor_buf_size]);
    auto allocated_buffer = ttnn::allocate_buffer_on_device(tensor_buf_size * datum_size_bytes, device, tensor_shape, DataType::BFLOAT16, Layout::TILE, mem_cfg);
    auto allocated_storage = tt::tt_metal::DeviceStorage{allocated_buffer};
    auto allocated_tensor = Tensor(allocated_storage, tensor_shape, DataType::BFLOAT16, Layout::TILE);
    auto readback_data = std::shared_ptr<bfloat16 []>(new bfloat16[tensor_buf_size]);

    std::thread t0([&] () {
        for (int j = 0; j < 1000; j++) {
            if (j != 0) {
                ttnn::event_synchronize(read_event);
            }
            read_event = std::make_shared<Event>();
            for (int i = 0; i < tensor_buf_size; i++) {
                host_data[i] = bfloat16(static_cast<float>(2 + j));
            }
            ttnn::write_buffer(write_cq, allocated_tensor, {host_data});
            ttnn::record_event(device->command_queue(write_cq), write_event);
        }
    });

    std::thread t1([&] () {
        for (int j = 0; j < 1000; j++) {
            ttnn::event_synchronize(write_event);
            write_event = std::make_shared<Event>();
            ttnn::read_buffer(read_cq, allocated_tensor, {readback_data});
            for (int i = 0; i < tensor_buf_size; i++) {
                EXPECT_EQ(readback_data[i], host_data[i]);
            }
            ttnn::record_event(device->command_queue(read_cq), read_event);
        }
    });

    t0.join();
    t1.join();
}
