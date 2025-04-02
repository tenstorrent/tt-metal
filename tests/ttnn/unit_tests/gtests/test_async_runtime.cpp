// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/move/utility_core.hpp>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/event.hpp>
#include <cmath>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/logger.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/small_vector.hpp>
#include "strong_type.hpp"
#include "ttnn/async_runtime.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/cpp/ttnn/operations/creation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/moreh/moreh_sum/moreh_sum.hpp"
#include "ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/host_buffer/owned_buffer.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace tt::tt_metal {
namespace {

using MultiCommandQueueSingleDeviceFixture = ::ttnn::MultiCommandQueueSingleDeviceFixture;

TEST_F(MultiCommandQueueSingleDeviceFixture, TestAsyncPreallocatedOutputs) {
    IDevice* device = this->device_;
    MemoryConfig mem_cfg = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        .buffer_type = BufferType::DRAM,
        .shard_spec = std::nullopt};

    uint32_t input_buf_size_datums = 1024 * 1024;
    uint32_t output_buf_size_datums = 1024 * 32;
    uint32_t datum_size_bytes = 2;
    ttnn::QueueId io_cq = ttnn::QueueId(1);                 // Data reads and writes done through CQ0
    ttnn::QueueId workload_dispatch_cq = ttnn::QueueId(0);  // Workload dispatched through CQ1

    ttnn::Shape input_shape({1, 1, 1024, 1024});
    auto host_data = std::shared_ptr<bfloat16[]>(new bfloat16[input_buf_size_datums]);
    auto readback_data = std::shared_ptr<bfloat16[]>(new bfloat16[output_buf_size_datums]);

    for (int i = 0; i < input_buf_size_datums; i++) {
        host_data[i] = bfloat16(static_cast<float>(1));
    }
    // Create golden data using tt_eager APIs
    Tensor np_tensor = ttnn::full(input_shape, static_cast<float>(1), DataType::BFLOAT16, Layout::TILE, *device_);
    ttnn::SmallVector<int64_t> reduce_dims = {3};
    Tensor np_out = ttnn::moreh_sum(np_tensor, reduce_dims, false, std::nullopt, std::nullopt, std::nullopt);
    Tensor np_out_host = np_out.cpu();
    const bfloat16* golden_output =
        std::get<owned_buffer::Buffer<bfloat16>>(std::get<OwnedStorage>(np_out_host.get_storage()).buffer).begin();
    // Enable Asynchronous Execution and test ttnn runtime APIs
    device_->enable_async(true);
    // Events for host - device synchronization
    auto write_event = std::make_shared<Event>();
    auto workload_event = std::make_shared<Event>();
    // Running sum-reduce with preallocated output
    // Preallocate Input and Output Tensors on Device
    tt_metal::TensorLayout tensor_layout(DataType::BFLOAT16, PageConfig(Layout::TILE), mem_cfg);
    ASSERT_EQ(input_buf_size_datums * datum_size_bytes, tensor_layout.compute_packed_buffer_size_bytes(input_shape));
    ASSERT_EQ(
        output_buf_size_datums * datum_size_bytes,
        tensor_layout.compute_packed_buffer_size_bytes(np_out.get_padded_shape()));
    auto input_buffer =
        tt::tt_metal::tensor_impl::allocate_buffer_on_device(device_, TensorSpec(input_shape, tensor_layout));
    auto output_buffer = tt::tt_metal::tensor_impl::allocate_buffer_on_device(
        device_, TensorSpec(np_out.get_padded_shape(), tensor_layout));
    auto input_storage = tt::tt_metal::DeviceStorage{input_buffer};
    auto output_storage = tt::tt_metal::DeviceStorage{output_buffer};
    Tensor input_tensor = Tensor(
        input_storage,
        TensorSpec(input_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), MemoryConfig{})));
    Tensor output_tensor = Tensor(output_storage, np_out.get_logical_shape(), DataType::BFLOAT16, Layout::TILE);
    // Populate input_tensor with data
    ttnn::write_buffer(io_cq, input_tensor, {host_data});
    // Record the completion of the write event
    ttnn::record_event(device_->command_queue(*io_cq), write_event);
    // Host stalls until write is completed, before sending workload
    ttnn::event_synchronize(write_event);
    EXPECT_EQ(ttnn::event_query(write_event), true);
    // Dispatch workload. Preallocated output_tensor is populated by op/
    ttnn::moreh_sum(input_tensor, /*dim*/ 3, false, output_tensor, std::nullopt, std::nullopt);
    // Record completion of workload
    ttnn::record_event(device_->command_queue(*workload_dispatch_cq), workload_event);
    ttnn::event_synchronize(workload_event);
    EXPECT_EQ(ttnn::event_query(workload_event), true);
    // Read output back, once workload is complete
    ttnn::read_buffer(io_cq, output_tensor, {readback_data});
    // Buffers are currently jointly owned by the original buffer object, the storage object and the tensor (3).
    EXPECT_EQ(input_buffer.use_count(), 3);
    EXPECT_EQ(output_buffer.use_count(), 3);
    // Deallocate tensors (tensor gives up buffer). Done asynchronously, so sync on queue after.
    input_tensor.deallocate();
    output_tensor.deallocate();
    ttnn::queue_synchronize(device_->command_queue(*io_cq));
    // Buffer only has 2 owners in main thread.
    EXPECT_EQ(input_buffer.use_count(), 2);
    EXPECT_EQ(output_buffer.use_count(), 2);
    for (int i = 0; i < output_buf_size_datums; i++) {
        EXPECT_EQ(readback_data[i], golden_output[i]);
    }
}

TEST_F(MultiCommandQueueSingleDeviceFixture, TestAsyncRuntimeAllocatedBuffers) {
    device_->enable_async(true);
    MemoryConfig mem_cfg = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        .buffer_type = BufferType::DRAM,
        .shard_spec = std::nullopt};

    uint32_t buf_size_datums = 1024 * 1024;
    uint32_t datum_size_bytes = 2;
    std::vector<uint32_t> inputs = {4, 9, 16, 25, 36, 64};
    ttnn::QueueId io_cq = ttnn::QueueId(1);
    ttnn::QueueId workload_dispatch_cq = ttnn::QueueId(0);
    ttnn::Shape shape{1, 1, 1024, 1024};

    auto host_data = std::shared_ptr<bfloat16[]>(new bfloat16[buf_size_datums]);
    auto readback_data = std::shared_ptr<bfloat16[]>(new bfloat16[buf_size_datums]);
    for (int loop = 0; loop < 10; loop++) {
        log_info(LogTest, "Running outer loop {}", loop);
        for (auto input_val : inputs) {
            for (int i = 0; i < buf_size_datums; i++) {
                host_data[i] = bfloat16(static_cast<float>(input_val));
            }

            auto write_event = std::make_shared<Event>();
            auto workload_event = std::make_shared<Event>();
            TensorLayout tensor_layout(DataType::BFLOAT16, PageConfig(Layout::TILE), mem_cfg);
            ASSERT_EQ(buf_size_datums * datum_size_bytes, tensor_layout.compute_packed_buffer_size_bytes(shape));
            auto input_buffer =
                tt::tt_metal::tensor_impl::allocate_buffer_on_device(device_, TensorSpec(shape, tensor_layout));
            auto input_storage = tt::tt_metal::DeviceStorage{input_buffer};
            Tensor input_tensor = Tensor(input_storage, shape, DataType::BFLOAT16, Layout::TILE);
            ttnn::write_buffer(io_cq, input_tensor, {host_data});            // Write using cq 1
            ttnn::record_event(device_->command_queue(*io_cq), write_event);  // Record write on cq 1
            // Wait until cq 1 write is complete
            ttnn::wait_for_event(device_->command_queue(*workload_dispatch_cq), write_event);

            // Run operation on cq 0
            Tensor output_tensor = ttnn::sqrt(workload_dispatch_cq, input_tensor);
            auto dummy_buffer_0 =
                tt::tt_metal::tensor_impl::allocate_buffer_on_device(device_, TensorSpec(shape, tensor_layout));
            output_tensor = ttnn::neg(workload_dispatch_cq, output_tensor);
            // Allocate this buffer to stress test async allocation across op execution and explicit allocation
            auto dummy_buffer_1 =
                tt::tt_metal::tensor_impl::allocate_buffer_on_device(device_, TensorSpec(shape, tensor_layout));
            // Record cq 0 prog execution
            ttnn::record_event(device_->command_queue(*workload_dispatch_cq), workload_event);
            // Wait until cq 0 prog execution is done
            ttnn::wait_for_event(device_->command_queue(*io_cq), workload_event);
            // Read using cq 1
            ttnn::read_buffer(io_cq, output_tensor, {readback_data});
            for (int i = 0; i < buf_size_datums; i++) {
                EXPECT_EQ(
                    static_cast<int>(std::floor(bfloat16(readback_data[i]).to_float())),
                    static_cast<int>(-1 * sqrt(input_val)));
            }
        }
    }
}

TEST_F(MultiCommandQueueSingleDeviceFixture, TestAsyncRuntimeBufferDestructor) {
    // Test functionality for the buffer destructor, which will call deallocate asynchronously
    // We must ensure that the deallocate step, which can run after the buffer has been destroyed
    // does not rely on stale buffer state, after the buffer has been destroyed on host
    device_->enable_async(true);
    MemoryConfig mem_cfg = MemoryConfig{
        .memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        .buffer_type = BufferType::DRAM,
        .shard_spec = std::nullopt};

    uint32_t buf_size_datums = 1024 * 1024;
    uint32_t datum_size_bytes = 2;
    ttnn::Shape shape{1, 1, 1024, 1024};
    // Inside the loop, initialize a buffer with limited lifetime.
    // This will asynchronously allocate the buffer, wait for the allocation to complete (address to be assigned to the
    // buffer), destroy the buffer (which will asynchronously deallocate the buffer) in a loop
    TensorLayout tensor_layout(DataType::BFLOAT16, PageConfig(Layout::TILE), mem_cfg);
    TensorSpec tensor_spec(shape, tensor_layout);
    for (int loop = 0; loop < 100000; loop++) {
        auto input_buffer_dummy = tt::tt_metal::tensor_impl::allocate_buffer_on_device(device_, tensor_spec);
        device_->synchronize();
    }
}
}  // namespace
}  // namespace tt::tt_metal
