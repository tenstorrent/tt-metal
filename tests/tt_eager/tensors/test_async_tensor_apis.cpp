// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <boost/move/utility_core.hpp>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/logger.hpp>
#include <tt-metalium/shape.hpp>
#include "tests/tt_metal/tt_metal/common/dispatch_fixture.hpp"
#include "ttnn/cpp/ttnn/operations/creation.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/reshape/view.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace tt::tt_metal {
namespace {

using ::tt::constants::TILE_HEIGHT;
using ::tt::constants::TILE_WIDTH;

uint32_t get_device_buffer_address(const Tensor& tensor) {
    TT_FATAL(std::holds_alternative<DeviceStorage>(tensor.get_storage()), "Tensor storage is not DeviceStorage");
    auto buffer = std::get<DeviceStorage>(tensor.get_storage()).buffer;
    uint32_t result = 0;
    buffer->device()->push_work([&]() { result = buffer->address(); }, true);
    return result;
}

Tensor tensor_identity_copy_function(const Tensor& tensor) { return tensor; }

TEST_F(DispatchFixture, TestAsyncRefCountManager) {
    IDevice* device = this->devices_[0];

    log_info(LogTest, "Testing Device tensor copy assignment");
    for (int i = 0; i < 5; i++) {
        // Run for multiple loops to ensure deterministic behaviour with device addresses
        // Initialize 2 tensors on device
        Tensor tensor1 = ttnn::full(
            ttnn::Shape({1, 1, 1024, 1024}),
            static_cast<float>(i),
            DataType::BFLOAT16,
            /*layout=*/std::nullopt,
            *device);
        Tensor tensor2 = ttnn::full(
            ttnn::Shape({1, 1, 1024, 1024}),
            static_cast<float>(i),
            DataType::BFLOAT16,
            /*layout=*/std::nullopt,
            *device);
        uint32_t tensor2_device_buf_addr = get_device_buffer_address(tensor2);
        tensor2 = tensor1;
        // To check if tensor2 is deallocated, create a third tensor on device and ensure that its address matches the
        // prev addr for tensor2
        Tensor tensor3 = ttnn::full(
            ttnn::Shape({1, 1, 1024, 1024}),
            static_cast<float>(i),
            DataType::BFLOAT16,
            /*layout=*/std::nullopt,
            *device);
        EXPECT_EQ(get_device_buffer_address(tensor3), tensor2_device_buf_addr);
        EXPECT_EQ(get_device_buffer_address(tensor1), get_device_buffer_address(tensor2));
    }
    log_info(LogTest, "Testing Device tensor self-assignment through function");
    for (int i = 0; i < 5; i++) {
        Tensor device_tensor = ttnn::full(
            ttnn::Shape({1, 1, 1024, 1024}),
            static_cast<float>(i),
            DataType::BFLOAT16,
            /*layout=*/std::nullopt,
            *device);
        uint32_t device_tensor_address = get_device_buffer_address(device_tensor);
        // This step will copy the tensor to a temp rval and std::move it back to the caller's instance of device_tensor
        // Ensure ref count and address remain unchanged
        device_tensor = tensor_identity_copy_function(device_tensor);
        EXPECT_EQ(get_device_buffer_address(device_tensor), device_tensor_address);
    }

    log_info(LogTest, "Testing Device tensor move assignment");
    for (int i = 0; i < 5; i++) {
        Tensor tensor1 = ttnn::full(
            ttnn::Shape({1, 1, 1024, 1024}),
            static_cast<float>(i),
            DataType::BFLOAT16,
            /*layout=*/std::nullopt,
            *device);
        Tensor tensor2 = std::move(tensor1);
    }

    log_info(LogTest, "Testing Device tensor self-assignment");
    Tensor tensor_to_self_assign = ttnn::full(
        ttnn::Shape({1, 1, 1024, 1024}),
        static_cast<float>(0),
        DataType::BFLOAT16,
        /*layout=*/std::nullopt,
        *device);
    uint32_t tensor_to_self_assign_address = get_device_buffer_address(tensor_to_self_assign);
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#pragma clang diagnostic ignored "-Wself-move"
#endif
    tensor_to_self_assign = tensor_to_self_assign;
    tensor_to_self_assign = std::move(tensor_to_self_assign);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
    EXPECT_EQ(get_device_buffer_address(tensor_to_self_assign), tensor_to_self_assign_address);
    auto barrier_tensor = tensor_to_self_assign.cpu();
}

}  // namespace
}  // namespace tt::tt_metal
