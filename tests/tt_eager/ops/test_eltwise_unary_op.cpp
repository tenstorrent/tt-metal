// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>

#include "common/constants.hpp"
#include "tensor/owned_buffer.hpp"
#include "tensor/owned_buffer_functions.hpp"
#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_dnn/op_library/operation.hpp"
#include "tt_dnn/op_library/pad/pad_op.hpp"
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

Tensor gelu_fast(const Tensor& t) { return tt::tt_metal::gelu(t, true); }

Tensor gelu_slow(const Tensor& t) { return tt::tt_metal::gelu(t, false); }

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

template <auto UnaryOpType, typename... Args>
bool run_test(Device* device, const Shape& shape, float low, float high, Args... args) {
    auto input_tensor = tt::numpy::random::uniform(bfloat16(low), bfloat16(high), shape).to(Layout::TILE);

    if constexpr (UnaryOpType == tt::tt_metal::UnaryOpType::SQRT) {
        auto host_output = host_function<::detail::sqrt>(input_tensor);
        auto device_output = tt::tt_metal::sqrt(input_tensor.to(device)).cpu();
        return tt::numpy::allclose<bfloat16>(host_output, device_output, args...);
    } else if constexpr (UnaryOpType == tt::tt_metal::UnaryOpType::EXP) {
        auto host_output = host_function<::detail::exp>(input_tensor);
        auto device_output = tt::tt_metal::exp(input_tensor.to(device)).cpu();
        return tt::numpy::allclose<bfloat16>(host_output, device_output, args...);
    } else if constexpr (UnaryOpType == tt::tt_metal::UnaryOpType::RECIP) {
        auto host_output = host_function<::detail::recip>(input_tensor);
        auto device_output = tt::tt_metal::recip(input_tensor.to(device)).cpu();
        return tt::numpy::allclose<bfloat16>(host_output, device_output, args...);
    } else if constexpr (UnaryOpType == tt::tt_metal::UnaryOpType::GELU) {
        auto host_output = host_function<::detail::gelu>(input_tensor);
        auto device_output = tt::tt_metal::gelu(input_tensor.to(device)).cpu();
        return tt::numpy::allclose<bfloat16>(host_output, device_output, args...);
    } else if constexpr (UnaryOpType == tt::tt_metal::UnaryOpType::RELU) {
        auto host_output = host_function<::detail::relu>(input_tensor);
        auto device_output = tt::tt_metal::relu(input_tensor.to(device)).cpu();
        return tt::numpy::allclose<bfloat16>(host_output, device_output, args...);
    } else if constexpr (UnaryOpType == tt::tt_metal::UnaryOpType::SIGMOID) {
        auto host_output = host_function<::detail::sigmoid>(input_tensor);
        auto device_output = tt::tt_metal::sigmoid(input_tensor.to(device)).cpu();
        return tt::numpy::allclose<bfloat16>(host_output, device_output, args...);
    } else if constexpr (UnaryOpType == tt::tt_metal::UnaryOpType::LOG) {
        auto host_output = host_function<::detail::log>(input_tensor);
        auto device_output = tt::tt_metal::log(input_tensor.to(device)).cpu();
        return tt::numpy::allclose<bfloat16>(host_output, device_output, args...);
    } else if constexpr (UnaryOpType == tt::tt_metal::UnaryOpType::TANH) {
        auto host_output = host_function<::detail::tanh>(input_tensor);
        auto device_output = tt::tt_metal::tanh(input_tensor.to(device)).cpu();
        return tt::numpy::allclose<bfloat16>(host_output, device_output, args...);
    }
    TT_ASSERT(false, "Unsupported function");
    return false;
}

void test_operation_infrastructure() {
    tt::log_info(tt::LogTest, "Running {}", __func__);
    using namespace tt::tt_metal;

    int device_id = 0;
    auto device = tt::tt_metal::CreateDevice(device_id);

    auto shape = Shape{1, 1, TILE_HEIGHT, TILE_WIDTH};
    auto input_tensor = tt::numpy::random::uniform(bfloat16(0), bfloat16(1), shape).to(Layout::TILE).to(device);

    auto op = operation::DeviceOperation(EltwiseUnary{
        {tt::tt_metal::UnaryWithParam{tt::tt_metal::UnaryOpType::SQRT, std::nullopt}},
        MemoryConfig{.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}});

    auto program_hash = op.compute_program_hash({input_tensor}, {});
    TT_FATAL(program_hash == 13008201745214372940ULL, fmt::format("Actual value is {}", program_hash));

    auto profiler_info = op.create_profiler_info({input_tensor});
    TT_FATAL(
        profiler_info.preferred_name.value() == "tt::tt_metal::EltwiseUnary",
        fmt::format("Actual value is {}", profiler_info.preferred_name.value()));
    TT_FATAL(
        profiler_info.parallelization_strategy.value() == "UnaryOpParallelizationStrategy::SINGLE_CORE",
        fmt::format("Actual value is {}", profiler_info.parallelization_strategy.value()));

    TT_FATAL(tt::tt_metal::CloseDevice(device));
}

void test_shape_padding() {
    tt::log_info(tt::LogTest, "Running {}", __func__);

    int device_id = 0;
    auto device = tt::tt_metal::CreateDevice(device_id);
    tt::tt_metal::AutoFormat::SetDefaultDevice(device);

    auto input_shape = Shape{1, 1, 13, 18};
    auto padded_input_shape = Shape{1, 1, TILE_HEIGHT, TILE_WIDTH};
    auto input_tensor = tt::numpy::random::uniform(bfloat16(0), bfloat16(1), input_shape);

    auto padded_input_tensor =
        tt::tt_metal::operation::run(tt::tt_metal::PadOnHost{padded_input_shape, {0, 0, 0, 0}, 0}, {input_tensor})
            .at(0);
    padded_input_tensor = padded_input_tensor.to(Layout::TILE);
    padded_input_tensor = padded_input_tensor.to(device);
    auto output_tensor =
        tt::tt_metal::operation::run(
            tt::tt_metal::EltwiseUnary{
                {tt::tt_metal::UnaryWithParam{tt::tt_metal::UnaryOpType::SQRT, std::nullopt}},
                tt::tt_metal::MemoryConfig{.memory_layout = tt::tt_metal::TensorMemoryLayout::INTERLEAVED}},
            {padded_input_tensor})
            .at(0);
    output_tensor = output_tensor.cpu();

    auto output_shape = output_tensor.get_legacy_shape();
    TT_FATAL(output_shape == padded_input_shape);
    TT_FATAL(output_shape.without_padding() == input_shape);

    TT_FATAL(tt::tt_metal::CloseDevice(device));
}

namespace tt {
namespace tt_metal {
template <bool approx_value = false>
struct exp_with_param {
    static Tensor fn(const tt::tt_metal::Tensor& t) {
        return exp(t, approx_value, operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
    }
};
}  // namespace tt_metal
}  // namespace tt

void test_numerically() {
    tt::log_info(tt::LogTest, "Running {}", __func__);

    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;

    int device_id = 0;
    auto device = tt::tt_metal::CreateDevice(device_id);

    auto shape = Shape{1, 1, TILE_HEIGHT, TILE_WIDTH};
    {
        auto allclose = run_test<tt::tt_metal::UnaryOpType::SQRT>(device, shape, 0.0f, 1.0f, 1e-1f, 1e-5f);
        TT_FATAL(allclose);
    }
    {
        auto allclose = run_test<tt::tt_metal::UnaryOpType::EXP>(device, shape, -1.0f, 1.0f, 1e-1f, 1e-5f);
        TT_FATAL(allclose);
    }
    {
        auto allclose = run_test<tt::tt_metal::UnaryOpType::EXP>(device, shape, -1.0f, 1.0f, 1e-1f, 1e-5f);
        TT_FATAL(allclose);
    }
    {
        auto allclose = run_test<tt::tt_metal::UnaryOpType::RECIP>(device, shape, 1.0f, 10.0f, 1e-1f, 1e-5f);
        TT_FATAL(allclose);
    }
    {
        auto allclose = run_test<tt::tt_metal::UnaryOpType::GELU>(device, shape, 1.0f, 10.0f, 1e-1f, 1e-3f);
        TT_FATAL(allclose);
    }
    {
        auto allclose = run_test<tt::tt_metal::UnaryOpType::GELU>(device, shape, 1.0f, 10.0f, 1e-1f, 1e-3f);
        TT_FATAL(allclose);
    }

    {
        auto allclose = run_test<tt::tt_metal::UnaryOpType::RELU>(device, shape, -1.0f, 1.0f, 1e-1f, 1e-5f);
        TT_FATAL(allclose);
    }
    {
        auto allclose = run_test<tt::tt_metal::UnaryOpType::SIGMOID>(device, shape, -1.0f, 1.0f, 1e-1f, 1e-5f);
        TT_FATAL(allclose);
    }
    {
        auto allclose = run_test<tt::tt_metal::UnaryOpType::LOG>(device, shape, 0.0f, 1.0f, 1e-1f, 1e-2f);
        TT_FATAL(allclose);
    }
    {
        auto allclose = run_test<tt::tt_metal::UnaryOpType::TANH>(device, shape, -1.0f, 1.0f, 1e-1f, 1e-5f);
        TT_FATAL(allclose);
    }

    TT_FATAL(tt::tt_metal::CloseDevice(device));
}

void test_program_cache() {
    tt::log_info(tt::LogTest, "Running {}", __func__);

    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;

    int device_id = 0;
    auto device = tt::tt_metal::CreateDevice(device_id);

    auto run_tests = [&]() {
        // Program Cache Miss
        run_test<tt::tt_metal::UnaryOpType::SQRT>(device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Hit
        run_test<tt::tt_metal::UnaryOpType::SQRT>(device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Miss
        run_test<tt::tt_metal::UnaryOpType::SQRT>(device, {1, 1, 384, 4096}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Miss
        run_test<tt::tt_metal::UnaryOpType::EXP>(device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Hit
        run_test<tt::tt_metal::UnaryOpType::SQRT>(device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Allocate a tensor to show that the addresses aren't cached
        auto input_tensor =
            tt::numpy::random::uniform(bfloat16(0.0f), bfloat16(0.0f), {1, 1, 32, 32}).to(Layout::TILE).to(device);

        // Program Cache Hit
        run_test<tt::tt_metal::UnaryOpType::EXP>(device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Miss
        run_test<tt::tt_metal::UnaryOpType::GELU>(device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 1.0f, 10.0f, 1e-1f, 1e-3f);

        // Program Cache Miss
        run_test<tt::tt_metal::UnaryOpType::GELU>(device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 1.0f, 10.0f, 1e-1f, 1e-3f);

        // Program Cache Hit
        run_test<tt::tt_metal::UnaryOpType::SQRT>(device, {1, 1, TILE_HEIGHT, TILE_WIDTH}, 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Hit
        run_test<tt::tt_metal::UnaryOpType::SQRT>(device, {1, 1, 384, 4096}, 0.0f, 1.0f, 1e-1f, 1e-5f);
    };

    device->enable_program_cache();
    run_tests();

    TT_FATAL(
        device->num_program_cache_entries() == 4,
        "There are {} entries",
        device->num_program_cache_entries());

    device->disable_and_clear_program_cache();
    TT_FATAL(device->num_program_cache_entries() == 0);
    TT_FATAL(tt::tt_metal::CloseDevice(device));
}

int main(int argc, char** argv) {
    test_operation_infrastructure();
    test_shape_padding();
    test_numerically();
    test_program_cache();
    return 0;
}
