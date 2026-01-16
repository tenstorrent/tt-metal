// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <algorithm>
#include <cmath>
#include <numbers>

#include <tt_stl/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/shape.hpp>
#include "ttnn/decorators.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/eltwise/unary/device/unary_device_operation.hpp"
#include "ttnn/operations/eltwise/unary/device/unary_device_operation_types.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/device.hpp"

using tt::tt_metal::DataType;
using tt::tt_metal::distributed::MeshDevice;

using tt::tt_metal::Layout;
using tt::tt_metal::Tensor;

namespace detail {
float sqrt(float x) { return std::sqrt(x); }
float exp(float x) { return std::exp(x); }
float recip(float x) { return 1 / x; }
float gelu(float x) { return x * (0.5 * (1 + std::erf(x / std::numbers::sqrt2))); }
float relu(float x) { return std::max(0.0f, x); }
float sigmoid(float x) { return 1 / (1 + std::exp(-x)); }
float log(float x) { return std::log(x); }
float tanh(float x) { return std::tanh(x); }
}  // namespace detail

Tensor gelu_fast(const Tensor& t) { return ttnn::gelu(t, true); }

Tensor gelu_slow(const Tensor& t) { return ttnn::gelu(t, false); }

template <auto UnaryFunction>
Tensor host_function(const Tensor& input_tensor) {
    auto input_buffer = tt::tt_metal::host_buffer::get_as<bfloat16>(input_tensor);

    auto output_buffer = std::vector<bfloat16>(input_tensor.physical_volume());

    for (auto index = 0; index < output_buffer.size(); index++) {
        auto value = UnaryFunction(static_cast<float>(input_buffer[index]));
        output_buffer[index] = bfloat16(value);
    }

    return Tensor(
        tt::tt_metal::HostBuffer(std::move(output_buffer)),
        input_tensor.logical_shape(),
        input_tensor.dtype(),
        input_tensor.layout());
}

template <ttnn::operations::unary::UnaryOpType unary_op_type, typename... Args>
bool run_test(MeshDevice* device, const ttnn::Shape& shape, float low, float high, Args... args) {
    auto input_tensor = ttnn::random::uniform(bfloat16(low), bfloat16(high), shape).to_layout(Layout::TILE);

    using ttnn::operations::unary::UnaryOpType;
    using ttnn::operations::unary::UnaryWithParam;

    if constexpr (unary_op_type == UnaryOpType::SQRT) {
        auto host_output = host_function<::detail::sqrt>(input_tensor);
        auto device_output = ttnn::sqrt(input_tensor.to_device(device), /*fast_and_approximate_mode=*/false).cpu();
        return ttnn::allclose<bfloat16>(host_output, device_output, args...);
    } else if constexpr (unary_op_type == UnaryOpType::EXP) {
        auto host_output = host_function<::detail::exp>(input_tensor);
        auto device_output = ttnn::exp(input_tensor.to_device(device)).cpu();
        return ttnn::allclose<bfloat16>(host_output, device_output, args...);
    } else if constexpr (unary_op_type == UnaryOpType::RECIP) {
        auto host_output = host_function<::detail::recip>(input_tensor);
        auto device_output = ttnn::reciprocal(input_tensor.to_device(device)).cpu();
        return ttnn::allclose<bfloat16>(host_output, device_output, args...);
    } else if constexpr (unary_op_type == UnaryOpType::GELU) {
        auto host_output = host_function<::detail::gelu>(input_tensor);
        auto device_output = ttnn::gelu(input_tensor.to_device(device)).cpu();
        return ttnn::allclose<bfloat16>(host_output, device_output, args...);
    } else if constexpr (unary_op_type == UnaryOpType::RELU) {
        auto host_output = host_function<::detail::relu>(input_tensor);
        auto device_output = ttnn::relu(input_tensor.to_device(device)).cpu();
        return ttnn::allclose<bfloat16>(host_output, device_output, args...);
    } else if constexpr (unary_op_type == UnaryOpType::SIGMOID) {
        auto host_output = host_function<::detail::sigmoid>(input_tensor);
        auto device_output = ttnn::sigmoid(input_tensor.to_device(device)).cpu();
        return ttnn::allclose<bfloat16>(host_output, device_output, args...);
    } else if constexpr (unary_op_type == UnaryOpType::LOG) {
        auto host_output = host_function<::detail::log>(input_tensor);
        auto device_output = ttnn::log(input_tensor.to_device(device), /*fast_and_approximate_mode=*/true).cpu();
        return ttnn::allclose<bfloat16>(host_output, device_output, args...);
    } else if constexpr (unary_op_type == UnaryOpType::TANH) {
        auto host_output = host_function<::detail::tanh>(input_tensor);
        auto device_output = ttnn::tanh(input_tensor.to_device(device)).cpu();
        return ttnn::allclose<bfloat16>(host_output, device_output, args...);
    }
    TT_FATAL(false, "Unsupported function");
    return false;
}

void test_operation_infrastructure() {
    using namespace tt::constants;
    log_info(tt::LogTest, "Running {}", __func__);

    using ttnn::operations::unary::UnaryOpType;
    using ttnn::operations::unary::UnaryWithParam;

    int device_id = 0;
    auto device_owner = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
    auto* device = device_owner.get();

    auto shape = ttnn::Shape({1, 1, TILE_HEIGHT, TILE_WIDTH});
    auto input_tensor =
        ttnn::random::uniform(bfloat16(0), bfloat16(1), shape).to_layout(Layout::TILE).to_device(device);

    ttnn::operations::unary::operation_attributes_t op_args{
        {UnaryWithParam{UnaryOpType::SQRT}}, DataType::BFLOAT16, tt::tt_metal::MemoryConfig{}, false, false};
    ttnn::operations::unary::tensor_args_t tensor_args{input_tensor};
    auto program_hash = ttnn::operations::unary::UnaryDeviceOperation::compute_program_hash(op_args, tensor_args);
    TT_FATAL(program_hash == 3018574135764717736ULL, "Actual value is {}", program_hash);
}

void test_shape_padding() {
    using namespace tt::constants;
    log_info(tt::LogTest, "Running {}", __func__);

    using ttnn::operations::unary::UnaryOpType;
    using ttnn::operations::unary::UnaryWithParam;

    int device_id = 0;
    auto device_owner = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
    auto* device = device_owner.get();
    ttnn::SetDefaultDevice(device);

    ttnn::Shape input_shape({1, 1, 13, 18});
    tt::tt_metal::Array4D padded_input_shape = {1, 1, TILE_HEIGHT, TILE_WIDTH};
    auto input_tensor = ttnn::random::uniform(bfloat16(0), bfloat16(1), input_shape);

    auto padded_input_tensor = ttnn::pad(input_tensor, padded_input_shape, tt::tt_metal::Array4D({0, 0, 0, 0}), 0);

    padded_input_tensor = padded_input_tensor.to_layout(Layout::TILE);
    padded_input_tensor = padded_input_tensor.to_device(device);
    auto output_tensor = ttnn::sqrt(padded_input_tensor);
    output_tensor = output_tensor.cpu();

    TT_FATAL(output_tensor.padded_shape() == padded_input_shape, "Error");
    TT_FATAL(output_tensor.logical_shape() == input_shape, "Error");
}

namespace tt::tt_metal {
template <bool approx_value = false>
struct exp_with_param {
    static Tensor fn(const tt::tt_metal::Tensor& t) {
        return ttnn::exp(t, approx_value, tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
    }
};
}  // namespace tt::tt_metal

void test_numerically() {
    log_info(tt::LogTest, "Running {}", __func__);

    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;
    using ttnn::operations::unary::UnaryOpType;
    using ttnn::operations::unary::UnaryWithParam;

    int device_id = 0;
    auto device_owner = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
    auto* device = device_owner.get();

    ttnn::Shape shape({1, 1, TILE_HEIGHT, TILE_WIDTH});
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
}

void test_program_cache() {
    log_info(tt::LogTest, "Running {}", __func__);

    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;
    using ttnn::operations::unary::UnaryOpType;
    using ttnn::operations::unary::UnaryWithParam;

    int device_id = 0;
    auto device_owner = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
    auto* device = device_owner.get();

    auto run_tests = [&]() {
        // Program Cache Miss
        run_test<UnaryOpType::SQRT>(device, ttnn::Shape({1, 1, TILE_HEIGHT, TILE_WIDTH}), 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Hit
        run_test<UnaryOpType::SQRT>(device, ttnn::Shape({1, 1, TILE_HEIGHT, TILE_WIDTH}), 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Miss
        run_test<UnaryOpType::SQRT>(device, ttnn::Shape({1, 1, 384, 4096}), 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Miss
        run_test<UnaryOpType::EXP>(device, ttnn::Shape({1, 1, TILE_HEIGHT, TILE_WIDTH}), 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Hit
        run_test<UnaryOpType::SQRT>(device, ttnn::Shape({1, 1, TILE_HEIGHT, TILE_WIDTH}), 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Allocate a tensor to show that the addresses aren't cached
        auto input_tensor = ttnn::random::uniform(bfloat16(0.0f), bfloat16(0.0f), ttnn::Shape({1, 1, 32, 32}))
                                .to_layout(Layout::TILE)
                                .to_device(device);

        // Program Cache Hit
        run_test<UnaryOpType::EXP>(device, ttnn::Shape({1, 1, TILE_HEIGHT, TILE_WIDTH}), 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Miss
        run_test<UnaryOpType::GELU>(device, ttnn::Shape({1, 1, TILE_HEIGHT, TILE_WIDTH}), 1.0f, 10.0f, 1e-1f, 1e-3f);

        // Program Cache Miss
        run_test<UnaryOpType::GELU>(device, ttnn::Shape({1, 1, TILE_HEIGHT, TILE_WIDTH}), 1.0f, 10.0f, 1e-1f, 1e-3f);

        // Program Cache Hit
        run_test<UnaryOpType::SQRT>(device, ttnn::Shape({1, 1, TILE_HEIGHT, TILE_WIDTH}), 0.0f, 1.0f, 1e-1f, 1e-5f);

        // Program Cache Hit
        run_test<UnaryOpType::SQRT>(device, ttnn::Shape({1, 1, 384, 4096}), 0.0f, 1.0f, 1e-1f, 1e-5f);
    };

    run_tests();

    TT_FATAL(device->num_program_cache_entries() == 4, "There are {} entries", device->num_program_cache_entries());

    device->disable_and_clear_program_cache();
    TT_FATAL(device->num_program_cache_entries() == 0, "Error");
}

int main() {
    // test_operation_infrastructure();
    test_shape_padding();
    test_numerically();
    test_program_cache();
    return 0;
}
