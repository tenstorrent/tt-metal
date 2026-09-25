// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "polynorm_fw_device_operation.hpp"

#include <enchantum/enchantum.hpp>
#include <string_view>
#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::polynorm3_fw::device {

namespace {

void validate_tensor(
    const ttnn::Tensor& tensor,
    std::string_view name,
    tt::tt_metal::DataType expected_dtype,
    const ttnn::Tensor& input) {
    TT_FATAL(
        tensor.storage_type() == ttnn::StorageType::DEVICE,
        "PolyNorm3Forward: {} must be on Device. Storage type: {}",
        name,
        enchantum::to_string(tensor.storage_type()));
    TT_FATAL(tensor.buffer() != nullptr, "PolyNorm3Forward: {} must have an allocated device buffer", name);
    TT_FATAL(tensor.device() == input.device(), "PolyNorm3Forward: {} must be on the same device as input", name);
    TT_FATAL(
        tensor.layout() == tt::tt_metal::Layout::TILE,
        "PolyNorm3Forward: {} must use TILE layout. Layout: {}",
        name,
        enchantum::to_string(tensor.layout()));
    TT_FATAL(
        tensor.dtype() == expected_dtype,
        "PolyNorm3Forward: {} must use {}. Data type: {}",
        name,
        enchantum::to_string(expected_dtype),
        enchantum::to_string(tensor.dtype()));
    TT_FATAL(
        tensor.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
        "PolyNorm3Forward: {} must use INTERLEAVED memory layout. Memory layout: {}",
        name,
        enchantum::to_string(tensor.memory_config().memory_layout()));
    TT_FATAL(
        tensor.buffer()->buffer_type() == ttnn::BufferType::DRAM,
        "PolyNorm3Forward: {} must be in DRAM. Buffer type: {}",
        name,
        enchantum::to_string(tensor.buffer()->buffer_type()));

    const auto tile = tensor.tensor_spec().tile();
    const auto face_shape = tile.get_face_shape();
    TT_FATAL(
        tile.get_height() == tt::constants::TILE_HEIGHT && tile.get_width() == tt::constants::TILE_WIDTH &&
            face_shape[0] == tt::constants::FACE_HEIGHT && face_shape[1] == tt::constants::FACE_WIDTH &&
            tile.get_num_faces() == tt::constants::TILE_HW / tt::constants::FACE_HW &&
            !tile.get_transpose_within_face() && !tile.get_transpose_of_faces(),
        "PolyNorm3Forward: {} must use the canonical untransposed 32x32 tile with 16x16 faces",
        name);
}

tt::tt_metal::TensorSpec canonical_output_spec(const ttnn::Tensor& input) {
    return {
        input.logical_shape(),
        tt::tt_metal::TensorLayout(input.dtype(), tt::tt_metal::Layout::TILE, input.memory_config())};
}

}  // namespace

void PolyNorm3ForwardDeviceOperation::validate_on_program_cache_miss(
    const PolyNorm3FWAttributes&, const PolyNorm3FWTensorArgs& tensor_args) {
    const auto& input = tensor_args.input;
    validate_tensor(input, "input", ttnn::DataType::BFLOAT16, input);
    TT_FATAL(input.logical_shape().rank() == 4U, "PolyNorm3Forward: input must have rank 4");
    const auto input_shape = input.logical_shape().to_array_4D();
    TT_FATAL(
        input_shape[0] > 0U && input_shape[1] > 0U && input_shape[2] > 0U && input_shape[3] > 0U,
        "PolyNorm3Forward: all input dimensions must be positive. Shape: {}",
        input.logical_shape());
    TT_FATAL(
        input_shape[3] % tt::constants::TILE_WIDTH == 0U,
        "PolyNorm3Forward: input channels must be divisible by {}. Shape: {}",
        tt::constants::TILE_WIDTH,
        input.logical_shape());

    const auto expected_output_spec = canonical_output_spec(input);
    TT_FATAL(
        input.padded_shape() == expected_output_spec.padded_shape(),
        "PolyNorm3Forward: input padded shape {} must use canonical tile padding {}",
        input.padded_shape(),
        expected_output_spec.padded_shape());

    validate_tensor(tensor_args.weight, "weight", ttnn::DataType::BFLOAT16, input);
    TT_FATAL(
        tensor_args.weight.logical_shape() == ttnn::Shape({1U, 1U, 1U, 3U}),
        "PolyNorm3Forward: weight must have shape [1, 1, 1, 3]. Shape: {}",
        tensor_args.weight.logical_shape());
    validate_tensor(tensor_args.bias, "bias", ttnn::DataType::BFLOAT16, input);
    TT_FATAL(
        tensor_args.bias.logical_shape() == ttnn::Shape({1U, 1U, 1U, 1U}),
        "PolyNorm3Forward: bias must have shape [1, 1, 1, 1]. Shape: {}",
        tensor_args.bias.logical_shape());

    if (tensor_args.preallocated_output.has_value()) {
        const auto& output = tensor_args.preallocated_output.value();
        validate_tensor(output, "preallocated output", ttnn::DataType::BFLOAT16, input);
        TT_FATAL(
            output.tensor_spec() == expected_output_spec,
            "PolyNorm3Forward: preallocated output spec must exactly match the canonical input-derived output spec");
    }
}

PolyNorm3FWSpecReturn PolyNorm3ForwardDeviceOperation::compute_output_specs(
    const PolyNorm3FWAttributes&, const PolyNorm3FWTensorArgs& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return {tensor_args.preallocated_output->tensor_spec()};
    }
    return {tt::tt_metal::TensorSpec(
        tensor_args.input.logical_shape(),
        tt::tt_metal::TensorLayout(
            tensor_args.input.dtype(), tt::tt_metal::Layout::TILE, tensor_args.input.memory_config()))};
}

PolyNorm3FWTensorReturn PolyNorm3ForwardDeviceOperation::create_output_tensors(
    const PolyNorm3FWAttributes& op_attrs, const PolyNorm3FWTensorArgs& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    auto specs = compute_output_specs(op_attrs, tensor_args);
    return ttnn::create_device_tensor(specs[0], tensor_args.input.device());
}

}  // namespace ttml::metal::ops::polynorm3_fw::device

namespace ttnn::prim {

ttml::metal::ops::polynorm3_fw::device::PolyNorm3ForwardDeviceOperation::tensor_return_value_t ttml_polynorm3_fw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight,
    const ttnn::Tensor& bias,
    float epsilon,
    const std::optional<ttnn::Tensor>& preallocated_output) {
    using OperationType = ttml::metal::ops::polynorm3_fw::device::PolyNorm3ForwardDeviceOperation;

    const auto operation_attributes = OperationType::operation_attributes_t{
        .epsilon = epsilon,
    };
    const auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .weight = weight,
        .bias = bias,
        .preallocated_output = preallocated_output,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
