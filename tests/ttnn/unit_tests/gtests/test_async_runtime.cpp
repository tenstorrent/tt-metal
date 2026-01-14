// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/move/utility_core.hpp>
#include <cstring>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/event.hpp>
#include <cmath>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/shape.hpp>
#include <tt_stl/small_vector.hpp>
#include <tt_stl/strong_type.hpp>
#include "ttnn/async_runtime.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/decorators.hpp"
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
