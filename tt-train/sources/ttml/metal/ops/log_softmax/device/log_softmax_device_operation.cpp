// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "log_softmax_device_operation.hpp"

#include <optional>

#include "metal/common/tensor_validation.hpp"
#include "row_softmax_program.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::log_softmax::device {

namespace {
// Row state: maximum, selected index and log1p(tail), one FP32 tile each per tile row of the input.
constexpr uint32_t kStateTiles = 3;

tt::tt_metal::TensorSpec state_spec(const ttnn::Tensor& input) {
    const auto& tile = input.tensor_spec().tile();
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({1, 1, row_softmax::layout_for(input).rows * tile.get_height(), kStateTiles * tile.get_width()}),
        tt::tt_metal::TensorLayout(
            ttnn::DataType::FLOAT32, tt::tt_metal::PageConfig(ttnn::Layout::TILE, tile), ttnn::DRAM_MEMORY_CONFIG));
}
}  // namespace

void LogSoftmaxDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    check_device_tensor(
        input,
        "LogSoftmax",
        "input",
        {.dtypes = {ttnn::DataType::BFLOAT16, ttnn::DataType::FLOAT32},
         .layout = ttnn::Layout::TILE,
         .buffer_type = ttnn::BufferType::DRAM});
    TT_FATAL(input.logical_shape().rank() == 4, "LogSoftmax: expects rank-4 input, got {}", input.logical_shape());
    TT_FATAL(
        row_softmax::supports_tile(input.tensor_spec().tile()),
        "LogSoftmax: unsupported tile {}",
        input.tensor_spec().tile().get_tile_shape());
    TT_FATAL(input.memory_config() == ttnn::DRAM_MEMORY_CONFIG, "LogSoftmax: expects interleaved DRAM input");
    // The kernels take tile counts from the padded shape and mask only the last tile's columns past the logical width.
    check_tile_padding(input, "LogSoftmax", "input");
    TT_FATAL(
        input.logical_shape()[-1] > 0 && input.logical_shape()[-1] <= (1U << 24),
        "LogSoftmax: reduction width must be in [1, 2^24], got {}",
        input.logical_shape()[-1]);
    if (!args.backward) {
        return;
    }
    TT_FATAL(
        tensor_args.upstream.has_value() && tensor_args.state.has_value(),
        "LogSoftmax: backward requires the upstream gradient and the row state");
    const auto& upstream = *tensor_args.upstream;
    const auto& state = *tensor_args.state;
    check_device_tensor(
        upstream,
        "LogSoftmax",
        "upstream",
        {.dtypes = {ttnn::DataType::BFLOAT16, ttnn::DataType::FLOAT32}, .buffer_type = ttnn::BufferType::DRAM});
    check_device_tensor(
        state, "LogSoftmax", "state", {.dtypes = {ttnn::DataType::FLOAT32}, .buffer_type = ttnn::BufferType::DRAM});
    TT_FATAL(
        upstream.logical_shape() == input.logical_shape() && upstream.padded_shape() == input.padded_shape() &&
            upstream.memory_config() == input.memory_config() && upstream.layout() == input.layout() &&
            upstream.tensor_spec().tile() == input.tensor_spec().tile(),
        "LogSoftmax: upstream shape, layout and memory configuration must match the input (got {} in {}, expected "
        "{} in {})",
        upstream.logical_shape(),
        upstream.memory_config(),
        input.logical_shape(),
        input.memory_config());
    check_same_device(upstream, input, "LogSoftmax", "upstream", "input");
    check_same_device(state, input, "LogSoftmax", "state", "input");
    const auto expected_state = state_spec(input);
    TT_FATAL(
        state.tensor_spec() == expected_state,
        "LogSoftmax: invalid row state {}, expected FP32 TILE {}",
        state.logical_shape(),
        expected_state.logical_shape());
}

LogSoftmaxDeviceOperation::spec_return_value_t LogSoftmaxDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    spec_return_value_t specs{input.tensor_spec()};
    if (!args.backward) {
        specs.push_back(state_spec(input));
    }
    return specs;
}

LogSoftmaxDeviceOperation::tensor_return_value_t LogSoftmaxDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    tensor_return_value_t outputs;
    for (const auto& spec : compute_output_specs(args, tensor_args)) {
        outputs.push_back(ttnn::create_device_tensor(spec, tensor_args.input.device()));
    }
    return outputs;
}

ttsl::hash::hash_t LogSoftmaxDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // `backward` keys which optional tensors are present, so buffer bindings stay positional on a cache hit. Full
    // specs, because TensorAccessorArgs compile each buffer's placement into the kernels: an upstream gradient in
    // L1 must not reuse a reader built for DRAM.
    const auto spec_of = [](const std::optional<ttnn::Tensor>& t) {
        return t.has_value() ? std::optional<tt::tt_metal::TensorSpec>(t->tensor_spec()) : std::nullopt;
    };
    return tt::tt_metal::operation::hash_operation<LogSoftmaxDeviceOperation>(
        args, tensor_args.input.tensor_spec(), spec_of(tensor_args.upstream), spec_of(tensor_args.state));
}

}  // namespace ttml::metal::ops::log_softmax::device

namespace ttnn::prim {

ttml::metal::ops::log_softmax::device::LogSoftmaxDeviceOperation::tensor_return_value_t ttml_log_softmax_fw(
    const ttnn::Tensor& input_tensor) {
    using OperationType = ttml::metal::ops::log_softmax::device::LogSoftmaxDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.backward = false},
        OperationType::tensor_args_t{.input = input_tensor, .upstream = std::nullopt, .state = std::nullopt});
}

ttnn::Tensor ttml_log_softmax_bw(
    const ttnn::Tensor& input_tensor, const ttnn::Tensor& upstream_tensor, const ttnn::Tensor& state_tensor) {
    using OperationType = ttml::metal::ops::log_softmax::device::LogSoftmaxDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.backward = true},
        OperationType::tensor_args_t{.input = input_tensor, .upstream = upstream_tensor, .state = state_tensor})[0];
}

}  // namespace ttnn::prim
