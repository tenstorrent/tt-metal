// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>

#include "common/constants.hpp"
#include "queue/queue.hpp"
#include "tensor/owned_buffer.hpp"
#include "tensor/owned_buffer_functions.hpp"
#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_dnn/op_library/operation.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_numpy/functions.hpp"

using tt::tt_metal::DataType;
using tt::tt_metal::Device;

using tt::tt_metal::Layout;
using tt::tt_metal::OwnedStorage;
using tt::tt_metal::Shape;
using tt::tt_metal::Tensor;

namespace detail {
float sqrt(float x) { return std::sqrt(x); }
float exp(float x) { return std::exp(x); }
float recip(float x) { return 1 / x; }
float gelu(float x) { return x * (0.5 * (1 + std::erf(x / std::sqrt(2)))); }
float relu(float x) { return std::max(0.0f, x); }
float sigmoid(float x) { return 1 / (1 + std::exp(-x)); }
float log(float x) { return std::log(x); }
float tanh(float x) { return std::tanh(x); }
}  // namespace detail

template <auto UnaryFunction>
Tensor host_function(const Tensor& input_tensor) {
    auto input_buffer = tt::tt_metal::owned_buffer::get_as<bfloat16>(input_tensor);
    auto output_buffer = tt::tt_metal::owned_buffer::create<bfloat16>(input_tensor.volume());
    for (auto index = 0; index < output_buffer.size(); index++) {
        auto value = UnaryFunction(input_buffer[index].to_float());
        output_buffer[index] = bfloat16(value);
    }
    return Tensor(OwnedStorage{output_buffer}, input_tensor.get_legacy_shape(), input_tensor.get_dtype(), input_tensor.get_layout());
}

void test_multi_queue_api() {
    tt::log_info(tt::LogTest, "Running {}", __func__);

    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;

    int device_id = 0;
    auto num_command_queues = 1;
    auto device = tt::tt_metal::CreateDevice(device_id, num_command_queues);

    auto& command_queue_0 = device->command_queue(0);
    // auto& command_queue_1 = device->command_queue(1);

    auto host_input_tensor =
        tt::numpy::random::uniform(bfloat16(0.0f), bfloat16(1.0f), {1, 1, TILE_HEIGHT, TILE_WIDTH}).to(Layout::TILE);

    // Base API
    /*

        auto operation = tt::tt_metal::operation::DeviceOperation(
            tt::tt_metal::EltwiseUnary{std::vector{tt::tt_metal::UnaryWithParam{tt::tt_metal::UnaryOpType::SQRT}}});

        auto optional_input_tensors = std::vector<std::optional<const Tensor>>{};
        auto optional_output_tensors = std::vector<std::optional<Tensor>>{};

        auto cq0_input_tensor = host_input_tensor.to(command_queue_0);  // Device::CQ0
        auto cq0_input_tensors = std::vector{cq0_input_tensor};

        auto cq1_input_tensor = host_input_tensor.to(command_queue_1);  // Device::CQ1
        auto cq1_input_tensors = std::vector{cq1_input_tensor};

        auto cq0_output_tensors = tt::tt_metal::EnqueueOperation(
            command_queue_0, operation, cq0_input_tensors, optional_input_tensors, optional_output_tensors);

        // Can't use command_queue_1 for now. So, use command_queue_0 for both
        auto cq1_output_tensors = tt::tt_metal::EnqueueOperation(
            command_queue_0, operation, cq1_input_tensors, optional_input_tensors, optional_output_tensors);
    */
    // Base API END

    // OP API
    auto input_tensor = host_input_tensor.to(device);
    auto cq0_output_tensor = tt::tt_metal::sqrt(input_tensor);  // Default CQ (CQ0)

    // Can't use command_queue_1 for now. So, use command_queue_0 for both
    auto cq1_output_tensor = tt::tt_metal::sqrt(command_queue_0, input_tensor);  // CQ1
    // OP API END

    auto blocking = true;
    cq0_output_tensor = cq0_output_tensor.cpu(blocking);
    cq1_output_tensor = cq1_output_tensor.cpu(blocking);

    auto golden_output_tensor = host_function<::detail::sqrt>(host_input_tensor);

    tt::tt_metal::QueueSynchronize(command_queue_0);
    // tt::tt_metal::QueueSynchronize(command_queue_1);

    TT_FATAL(tt::numpy::allclose<bfloat16>(golden_output_tensor, cq0_output_tensor, 1e-3f, 1e-2f));
    TT_FATAL(tt::numpy::allclose<bfloat16>(golden_output_tensor, cq1_output_tensor, 1e-3f, 1e-2f));

    TT_FATAL(tt::tt_metal::CloseDevice(device));
}

int main(int argc, char** argv) {
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr) {
        test_multi_queue_api();
    }
    return 0;
}
