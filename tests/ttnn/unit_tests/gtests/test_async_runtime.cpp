// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/move/utility_core.hpp>
#include <cstring>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <tt-metalium/bfloat16.hpp>
#include <cmath>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/shape.hpp>
#include <tt_stl/small_vector.hpp>
#include <tt_stl/strong_type.hpp>
#include "ttnn/async_runtime.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/moreh/moreh_sum/moreh_sum.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
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
    auto* device = this->device_;
    MemoryConfig mem_cfg = MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};

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
    const Tensor reference_tensor = ttnn::distributed::get_device_tensors(np_out_host).front();
    auto golden_output = host_buffer::get_as<bfloat16>(reference_tensor);
    // Events for host - device synchronization
    // Running sum-reduce with preallocated output
    // Preallocate Input and Output Tensors on Device
    tt_metal::TensorLayout tensor_layout(DataType::BFLOAT16, PageConfig(Layout::TILE), mem_cfg);
    ASSERT_EQ(input_buf_size_datums * datum_size_bytes, tensor_layout.compute_packed_buffer_size_bytes(input_shape));
    ASSERT_EQ(
        output_buf_size_datums * datum_size_bytes,
        tensor_layout.compute_packed_buffer_size_bytes(np_out.padded_shape()));
    auto input_tensor = create_device_tensor(TensorSpec(input_shape, tensor_layout), device);
    auto output_tensor = create_device_tensor(TensorSpec(np_out.logical_shape(), tensor_layout), device);
    // Populate input_tensor with data
    ttnn::write_buffer(io_cq, input_tensor, {host_data});
    // Record the completion of the write event
    auto write_event = ttnn::record_event_to_host(device_->mesh_command_queue(*io_cq));
    // Host stalls until write is completed, before sending workload
    ttnn::event_synchronize(write_event);
    // Dispatch workload. Preallocated output_tensor is populated by op/
    ttnn::moreh_sum(input_tensor, /*dim*/ 3, false, output_tensor, std::nullopt, std::nullopt);
    // Record completion of workload
    auto workload_event = ttnn::record_event_to_host(device_->mesh_command_queue(*workload_dispatch_cq));
    ttnn::event_synchronize(workload_event);
    // Read output back, once workload is complete
    ttnn::read_buffer(io_cq, output_tensor, {readback_data});
    // Deallocate tensors (tensor gives up buffer). Done asynchronously, so sync on queue after.
    input_tensor.deallocate();
    output_tensor.deallocate();
    ttnn::queue_synchronize(device_->mesh_command_queue(*io_cq));
    EXPECT_EQ(std::memcmp(readback_data.get(), golden_output.data(), output_buf_size_datums * datum_size_bytes), 0);
}

TEST_F(MultiCommandQueueSingleDeviceFixture, TestAsyncRuntimeAllocatedBuffers) {
    MemoryConfig mem_cfg = MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};

    uint32_t buf_size_datums = 1024 * 1024;
    uint32_t datum_size_bytes = 2;
    std::vector<uint32_t> inputs = {4, 9, 16, 25, 36, 64};
    ttnn::QueueId io_cq = ttnn::QueueId(1);
    ttnn::QueueId workload_dispatch_cq = ttnn::QueueId(0);
    ttnn::Shape shape{1, 1, 1024, 1024};

    auto host_data = std::shared_ptr<bfloat16[]>(new bfloat16[buf_size_datums]);
    auto readback_data = std::shared_ptr<bfloat16[]>(new bfloat16[buf_size_datums]);
    auto expected_data = std::shared_ptr<bfloat16[]>(new bfloat16[buf_size_datums]);
    for (int loop = 0; loop < 10; loop++) {
        log_info(LogTest, "Running outer loop {}", loop);
        for (auto input_val : inputs) {
            for (uint32_t i = 0; i < buf_size_datums; i++) {
                host_data[i] = bfloat16(static_cast<float>(input_val));
            }
            // Pre-compute expected output: neg(sqrt(input_val)) = -sqrt(input_val)
            float expected_val = static_cast<float>(-1 * sqrt(input_val));
            for (uint32_t i = 0; i < buf_size_datums; i++) {
                expected_data[i] = bfloat16(expected_val);
            }

            TensorLayout tensor_layout(DataType::BFLOAT16, PageConfig(Layout::TILE), mem_cfg);
            ASSERT_EQ(buf_size_datums * datum_size_bytes, tensor_layout.compute_packed_buffer_size_bytes(shape));
            auto input_tensor = create_device_tensor(TensorSpec(shape, tensor_layout), device_);
            ttnn::write_buffer(io_cq, input_tensor, {host_data});            // Write using cq 1
            auto write_event = ttnn::record_event(device_->mesh_command_queue(*io_cq));  // Record write on cq 1
            // Wait until cq 1 write is complete
            ttnn::wait_for_event(device_->mesh_command_queue(*workload_dispatch_cq), write_event);

            // Run operation on cq 0
            Tensor output_tensor;
            ttnn::with_command_queue_id(workload_dispatch_cq, [&]() { output_tensor = ttnn::sqrt(input_tensor); });

            // ==== DEBUG tt-metal#43725 — strip before merge ====
            // Capture sqrt's output buffer (S) address NOW, before `neg` reassigns `output_tensor`.
            uint64_t dbg_S = static_cast<uint64_t>(output_tensor.buffer()->address());
            // ==== END DEBUG ====

            auto dummy_buffer_0 =
                tt::tt_metal::tensor_impl::allocate_device_buffer(device_, TensorSpec(shape, tensor_layout));

            ttnn::with_command_queue_id(workload_dispatch_cq, [&]() { output_tensor = ttnn::neg(output_tensor); });

            // Allocate this buffer to stress test async allocation across op execution and explicit allocation
            auto dummy_buffer_1 =
                tt::tt_metal::tensor_impl::allocate_device_buffer(device_, TensorSpec(shape, tensor_layout));
            // Record cq 0 prog execution
            auto workload_event = ttnn::record_event(device_->mesh_command_queue(*workload_dispatch_cq));
            // Wait until cq 0 prog execution is done
            ttnn::wait_for_event(device_->mesh_command_queue(*io_cq), workload_event);

            // ==== DEBUG tt-metal#43725 — strip before merge ====
            // Aliasing theory (Fable second opinion): does neg's fresh output buffer (N), allocated during
            // active dispatch on CQ0, end up aliasing sqrt's still-live output buffer (S)'s DRAM address?
            // Log every relevant buffer base address per inner iteration. `output_tensor` now holds N, and
            // the CQ1 read_buffer below sources from output_tensor.buffer(), so READ_SRC == N by construction
            // (logged separately in case any rebinding occurs). One greppable line per iteration; log_info
            // flushes so lines survive a later hang/crash.
            uint64_t dbg_input = static_cast<uint64_t>(input_tensor.buffer()->address());
            uint64_t dbg_D0 = static_cast<uint64_t>(dummy_buffer_0->address());
            uint64_t dbg_N = static_cast<uint64_t>(output_tensor.buffer()->address());
            uint64_t dbg_D1 = static_cast<uint64_t>(dummy_buffer_1->address());
            uint64_t dbg_read_src = static_cast<uint64_t>(output_tensor.buffer()->address());
            // ==== END DEBUG ====

            // Read using cq 1
            ttnn::read_buffer(io_cq, output_tensor, {readback_data});

            // ==== DEBUG tt-metal#43725 — strip before merge ====
            int dbg_cmp = std::memcmp(readback_data.get(), expected_data.get(), buf_size_datums * datum_size_bytes);
            log_info(
                LogTest,
                "DBG43725 loop={} input_val={} INPUT=0x{:x} S=0x{:x} D0=0x{:x} N=0x{:x} D1=0x{:x} READ_SRC=0x{:x} "
                "N_eq_S={} READSRC_eq_S={} memcmp={}",
                loop,
                input_val,
                dbg_input,
                dbg_S,
                dbg_D0,
                dbg_N,
                dbg_D1,
                dbg_read_src,
                (dbg_N == dbg_S) ? 1 : 0,
                (dbg_read_src == dbg_S) ? 1 : 0,
                dbg_cmp);
            if (dbg_cmp != 0) {
                log_error(
                    LogTest,
                    "DBG43725 FAIL loop={} input_val={} N_eq_S={} READSRC_eq_S={} S=0x{:x} N=0x{:x} READ_SRC=0x{:x}",
                    loop,
                    input_val,
                    (dbg_N == dbg_S) ? 1 : 0,
                    (dbg_read_src == dbg_S) ? 1 : 0,
                    dbg_S,
                    dbg_N,
                    dbg_read_src);
            }
            // ==== END DEBUG ====

            EXPECT_EQ(std::memcmp(readback_data.get(), expected_data.get(), buf_size_datums * datum_size_bytes), 0)
                << "Data mismatch at loop " << loop << " input_val " << input_val;
        }
    }
}

TEST_F(MultiCommandQueueSingleDeviceFixture, TestAsyncRuntimeBufferDestructor) {
    // Test functionality for the buffer destructor, which will call deallocate asynchronously
    // We must ensure that the deallocate step, which can run after the buffer has been destroyed
    // does not rely on stale buffer state, after the buffer has been destroyed on host
    MemoryConfig mem_cfg = MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};

    ttnn::Shape shape{1, 1, 1024, 1024};
    // Inside the loop, initialize a buffer with limited lifetime.
    // This will asynchronously allocate the buffer, wait for the allocation to complete (address to be assigned to the
    // buffer), destroy the buffer (which will asynchronously deallocate the buffer) in a loop
    TensorLayout tensor_layout(DataType::BFLOAT16, PageConfig(Layout::TILE), mem_cfg);
    TensorSpec tensor_spec(shape, tensor_layout);
    for (int loop = 0; loop < 100000; loop++) {
        auto input_buffer_dummy = tt::tt_metal::tensor_impl::allocate_device_buffer(device_, tensor_spec);
    }
}
}  // namespace
}  // namespace tt::tt_metal
