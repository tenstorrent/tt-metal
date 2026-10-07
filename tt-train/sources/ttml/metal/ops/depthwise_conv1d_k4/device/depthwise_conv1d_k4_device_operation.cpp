// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "depthwise_conv1d_k4_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "depthwise_conv1d_k4_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::depthwise_conv1d_k4::device {

namespace {

void check_device_tensor(
    const ttnn::Tensor& tensor, const std::string& name, tt::tt_metal::Layout layout, const ttnn::Tensor& reference) {
    TT_FATAL(
        tensor.storage_type() == ttnn::StorageType::DEVICE && tensor.buffer() != nullptr,
        "depthwise_conv1d_k4: {} must be allocated on device",
        name);
    TT_FATAL(tensor.device() == reference.device(), "depthwise_conv1d_k4: {} must be on the input's device", name);
    TT_FATAL(
        tensor.layout() == layout,
        "depthwise_conv1d_k4: {} must be {}, got {}",
        name,
        enchantum::to_string(layout),
        enchantum::to_string(tensor.layout()));
    TT_FATAL(
        tensor.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "depthwise_conv1d_k4: {} must be BFLOAT16, got {}",
        name,
        enchantum::to_string(tensor.dtype()));
    TT_FATAL(
        tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED &&
            tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
        "depthwise_conv1d_k4: {} must be DRAM interleaved",
        name);
}

}  // namespace

void DepthwiseConv1dK4DeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    check_device_tensor(input, "input", tt::tt_metal::Layout::ROW_MAJOR, input);

    const auto& shape = input.logical_shape();
    TT_FATAL(shape.rank() == 4 && shape[0] == 1 && shape[1] == 1, "depthwise_conv1d_k4: input must be [1, 1, T, C]");
    const uint32_t seq = shape[-2];
    const uint32_t channels = shape[-1];
    TT_FATAL(
        seq > 0 && seq % tt::constants::TILE_HEIGHT == 0,
        "depthwise_conv1d_k4: T must be a positive multiple of {}",
        tt::constants::TILE_HEIGHT);
    TT_FATAL(
        channels > 0 && channels % tt::constants::TILE_WIDTH == 0,
        "depthwise_conv1d_k4: C must be a positive multiple of {}",
        tt::constants::TILE_WIDTH);

    for (const auto& [tap, name] : std::array{
             std::pair{&tensor_args.tap0, "tap0"},
             std::pair{&tensor_args.tap1, "tap1"},
             std::pair{&tensor_args.tap2, "tap2"},
             std::pair{&tensor_args.tap3, "tap3"}}) {
        check_device_tensor(*tap, name, tt::tt_metal::Layout::TILE, input);
        TT_FATAL(
            tap->logical_shape()[-1] == channels && tap->logical_volume() == channels,
            "depthwise_conv1d_k4: {} must hold C = {} values in its last dim",
            name,
            channels);
    }

    if (tensor_args.silu_grad.has_value()) {
        const auto& grad = tensor_args.silu_grad.value();
        check_device_tensor(grad, "silu_grad", tt::tt_metal::Layout::TILE, input);
        TT_FATAL(grad.logical_shape() == shape, "depthwise_conv1d_k4: silu_grad must match the input shape");
    }
}

spec_return_value_t DepthwiseConv1dK4DeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return tt::tt_metal::TensorSpec(
        tensor_args.input.logical_shape(),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            tt::tt_metal::MemoryConfig{}));
}

tensor_return_value_t DepthwiseConv1dK4DeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return ttnn::create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

ttsl::hash::hash_t DepthwiseConv1dK4DeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<DepthwiseConv1dK4DeviceOperation>(
        args.anti_causal, tensor_args.silu_grad.has_value(), tensor_args.input.logical_shape());
}

}  // namespace ttml::metal::ops::depthwise_conv1d_k4::device

namespace ttnn::prim {

ttml::metal::ops::depthwise_conv1d_k4::device::DepthwiseConv1dK4DeviceOperation::tensor_return_value_t
ttml_depthwise_conv1d_k4(
    const ttnn::Tensor& input,
    const ttnn::Tensor& tap0,
    const ttnn::Tensor& tap1,
    const ttnn::Tensor& tap2,
    const ttnn::Tensor& tap3,
    bool anti_causal,
    const std::optional<ttnn::Tensor>& silu_grad) {
    using OperationType = ttml::metal::ops::depthwise_conv1d_k4::device::DepthwiseConv1dK4DeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{.anti_causal = anti_causal};
    auto tensor_args = OperationType::tensor_args_t{
        .input = input, .tap0 = tap0, .tap1 = tap1, .tap2 = tap2, .tap3 = tap3, .silu_grad = silu_grad};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
