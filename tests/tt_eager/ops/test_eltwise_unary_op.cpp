// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>

#include "common/constants.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/host_buffer/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/unary/device/unary_device_operation.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operation.hpp"
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

Tensor gelu_fast(const Tensor& t) { return ttnn::gelu(t, true); }

Tensor gelu_slow(const Tensor& t) { return ttnn::gelu(t, false); }

template <auto UnaryFunction>
Tensor host_function(const Tensor& input_tensor) {
    auto input_buffer = tt::tt_metal::owned_buffer::get_as<bfloat16>(input_tensor);

    auto output_buffer = tt::tt_metal::owned_buffer::create<bfloat16>(input_tensor.volume());

    for (auto index = 0; index < output_buffer.size(); index++) {
        auto value = UnaryFunction(input_buffer[index].to_float());
        output_buffer[index] = bfloat16(value);
    }

    return Tensor(
        OwnedStorage{output_buffer},
        input_tensor.get_legacy_shape(),
        input_tensor.get_dtype(),
        input_tensor.get_layout());
}

template <ttnn::operations::unary::UnaryOpType unary_op_type, typename... Args>
bool run_test(Device* device, const Shape& shape, float low, float high, Args... args) {
    auto input_tensor = tt::numpy::random::uniform(bfloat16(low), bfloat16(high), shape).to(Layout::TILE);

    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;

    if constexpr (unary_op_type == UnaryOpType::SQRT) {
        auto host_output = host_function<::detail::sqrt>(input_tensor);
        auto device_output = ttnn::sqrt(input_tensor.to(device)).cpu();
        return tt::numpy::allclose<bfloat16>(host_output, device_output, args...);
    } else if constexpr (unary_op_type == UnaryOpType::EXP) {
        auto host_output = host_function<::detail::exp>(input_tensor);
        auto device_output = ttnn::exp(input_tensor.to(device)).cpu();
        return tt::numpy::allclose<bfloat16>(host_output, device_output, args...);
    } else if constexpr (unary_op_type == UnaryOpType::RECIP) {
        auto host_output = host_function<::detail::recip>(input_tensor);
        auto device_output = ttnn::reciprocal(input_tensor.to(device)).cpu();
        return tt::numpy::allclose<bfloat16>(host_output, device_output, args...);
    } else if constexpr (unary_op_type == UnaryOpType::GELU) {
        auto host_output = host_function<::detail::gelu>(input_tensor);
        auto device_output = ttnn::gelu(input_tensor.to(device)).cpu();
        return tt::numpy::allclose<bfloat16>(host_output, device_output, args...);
    } else if constexpr (unary_op_type == UnaryOpType::RELU) {
        auto host_output = host_function<::detail::relu>(input_tensor);
        auto device_output = ttnn::relu(input_tensor.to(device)).cpu();
        return tt::numpy::allclose<bfloat16>(host_output, device_output, args...);
    } else if constexpr (unary_op_type == UnaryOpType::SIGMOID) {
        auto host_output = host_function<::detail::sigmoid>(input_tensor);
        auto device_output = ttnn::sigmoid(input_tensor.to(device)).cpu();
        return tt::numpy::allclose<bfloat16>(host_output, device_output, args...);
    } else if constexpr (unary_op_type == UnaryOpType::LOG) {
        auto host_output = host_function<::detail::log>(input_tensor);
        auto device_output = ttnn::log(input_tensor.to(device)).cpu();
        return tt::numpy::allclose<bfloat16>(host_output, device_output, args...);
    } else if constexpr (unary_op_type == UnaryOpType::TANH) {
        auto host_output = host_function<::detail::tanh>(input_tensor);
        auto device_output =ttnn::tanh(input_tensor.to(device)).cpu();
        return tt::numpy::allclose<bfloat16>(host_output, device_output, args...);
    }
    TT_ASSERT(false, "Unsupported function");
    return false;
}

void test_operation_infrastructure() {
    using namespace tt::constants;
    tt::log_info(tt::LogTest, "Running {}", __func__);

    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;

    int device_id = 0;
    auto device = tt::tt_metal::CreateDevice(device_id);

    auto shape = Shape{1, 1, TILE_HEIGHT, TILE_WIDTH};
    auto input_tensor = tt::numpy::random::uniform(bfloat16(0), bfloat16(1), shape).to(Layout::TILE).to(device);

    ttnn::operations::unary::operation_attributes_t op_args {
        {UnaryWithParam{UnaryOpType::SQRT}},
        DataType::BFLOAT16,
        MemoryConfig{.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED},
        false,
        false};
    ttnn::operations::unary::tensor_args_t tensor_args {input_tensor};
    auto program_hash = ttnn::operations::unary::UnaryDeviceOperation::compute_program_hash(op_args, tensor_args);
    TT_FATAL(program_hash == 3018574135764717736ULL, "Actual value is {}", program_hash);

    TT_FATAL(tt::tt_metal::CloseDevice(device), "Error");
}

void test_shape_padding() {
    using namespace tt::constants;
    tt::log_info(tt::LogTest, "Running {}", __func__);

    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;

    int device_id = 0;
    auto device = tt::tt_metal::CreateDevice(device_id);
    ttnn::operations::experimental::auto_format::AutoFormat::SetDefaultDevice(device);

    tt::tt_metal::Array4D input_shape = {1, 1, 13, 18};
    tt::tt_metal::Array4D padded_input_shape = {1, 1, TILE_HEIGHT, TILE_WIDTH};
    auto input_tensor = tt::numpy::random::uniform(bfloat16(0), bfloat16(1), input_shape);

    auto padded_input_tensor = ttnn::pad(input_tensor, padded_input_shape, tt::tt_metal::Array4D({0, 0, 0, 0}), 0);

    padded_input_tensor = padded_input_tensor.to(Layout::TILE);
    padded_input_tensor = padded_input_tensor.to(device);
    auto output_tensor = ttnn::sqrt(padded_input_tensor);
    output_tensor = output_tensor.cpu();

    auto output_shape = output_tensor.get_legacy_shape();
    TT_FATAL(output_shape == tt::tt_metal::Shape(padded_input_shape), "Error");
    TT_FATAL(output_shape.without_padding() == tt::tt_metal::Shape(input_shape), "Error");

    TT_FATAL(tt::tt_metal::CloseDevice(device), "Error");
}

namespace tt {
namespace tt_metal {
template <bool approx_value = false>
struct exp_with_param {
    static Tensor fn(const tt::tt_metal::Tensor& t) {
        return ttnn::exp(t, approx_value, operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
    }
};
}  // namespace tt_metal
}  // namespace tt

void test_numerically() {
    tt::log_info(tt::LogTest, "Running {}", __func__);

    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;
    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;

    int device_id = 0;
    auto device = tt::tt_metal::CreateDevice(device_id);

    auto shape = Shape{1, 1, TILE_HEIGHT, TILE_WIDTH};
    {
        auto allclose = run_test<UnaryOpType::SQRT>(device, shape, 0.0f, 1.0f, 1e-1f, 1e-5f);
        TT_FATAL(allclose, "Error");
    }
    {
        auto allclose = run_test<UnaryOpType::EXP>(device, shape, -1.0f, 1.0f, 1e-1f, 1e-5f);
        TT_FATAL(allclose, "Error");
    }
    {
        auto allclose = run_test<UnaryOpType::EXP>(device, shape, -1.0f, 1.0f, 1e-1f, 1e-5f);
        TT_FATAL(allclose, "Error");
    }
    {
        auto allclose = run_test<UnaryOpType::RECIP>(device, shape, 1.0f, 10.0f, 1e-1f, 1e-5f);
        TT_FATAL(allclose, "Error");
    }
    {
        auto allclose = run_test<UnaryOpType::GELU>(device, shape, 1.0f, 10.0f, 1e-1f, 1e-3f);
        TT_FATAL(allclose, "Error");
    }
    {
        auto allclose = run_test<UnaryOpType::GELU>(device, shape, 1.0f, 10.0f, 1e-1f, 1e-3f);
        TT_FATAL(allclose, "Error");
    }

    {
        auto allclose = run_test<UnaryOpType::RELU>(device, shape, -1.0f, 1.0f, 1e-1f, 1e-5f);
        TT_FATAL(allclose, "Error");
    }
    {
        auto allclose = run_test<UnaryOpType::SIGMOID>(device, shape, -1.0f, 1.0f, 1e-1f, 1e-5f);
        TT_FATAL(allclose, "Error");
    }
    {
        auto allclose = run_test<UnaryOpType::LOG>(device, shape, 0.0f, 1.0f, 1e-1f, 1e-2f);
        TT_FATAL(allclose, "Error");
    }
    {
        auto allclose = run_test<UnaryOpType::TANH>(device, shape, -1.0f, 1.0f, 1e-1f, 1e-5f);
        TT_FATAL(allclose, "Error");
    }

    TT_FATAL(tt::tt_metal::CloseDevice(device), "Error");
}

void test_program_cache() {
    tt::log_info(tt::LogTest, "Running {}", __func__);

    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;
    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;

    int device_id = 0;
    auto device = tt::tt_metal::CreateDevice(device_id);

    auto run_tests = [&]() {
        // Program Cache Miss
        run_test<UnaryOpType::SQRT>(device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Hit
        run_test<UnaryOpType::SQRT>(device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Miss
        run_test<UnaryOpType::SQRT>(device, {1, 1, 384, 4096}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Miss
        run_test<UnaryOpType::EXP>(device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Hit
        run_test<UnaryOpType::SQRT>(device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Allocate a tensor to show that the addresses aren't cached
        auto input_tensor =
            tt::numpy::random::uniform(bfloat16(0.0f), bfloat16(0.0f), {1, 1, 32, 32}).to(Layout::TILE).to(device);

        // Program Cache Hit
        run_test<UnaryOpType::EXP>(device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Miss
        run_test<UnaryOpType::GELU>(device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 1.0f, 10.0f, 1e-1f, 1e-3f);

        // Program Cache Miss
        run_test<UnaryOpType::GELU>(device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 1.0f, 10.0f, 1e-1f, 1e-3f);

        // Program Cache Hit
        run_test<UnaryOpType::SQRT>(device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Hit
        run_test<UnaryOpType::SQRT>(device, {1, 1, 384, 4096}, 0.0f, 1.0f, 1e-1f, 1e-5f);
    };

    device->enable_program_cache();
    run_tests();

    TT_FATAL(device->num_program_cache_entries() == 4, "There are {} entries", device->num_program_cache_entries());

    device->disable_and_clear_program_cache();
    TT_FATAL(device->num_program_cache_entries() == 0, "Error");
    TT_FATAL(tt::tt_metal::CloseDevice(device), "Error");
}

int main(int argc, char** argv) {
    //test_operation_infrastructure();
    test_shape_padding();
    test_numerically();
    test_program_cache();
    return 0;
}
