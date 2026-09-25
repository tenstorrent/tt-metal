// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "polynorm_bw_device_operation.hpp"

#include <enchantum/enchantum.hpp>
#include <string_view>
#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::polynorm3_bw::device {

namespace {

void validate_tensor(
    const ttnn::Tensor& tensor,
    std::string_view name,
    tt::tt_metal::DataType expected_dtype,
    const ttnn::Tensor& input) {
    TT_FATAL(
        tensor.storage_type() == ttnn::StorageType::DEVICE,
        "PolyNorm3Backward: {} must be on Device. Storage type: {}",
        name,
        enchantum::to_string(tensor.storage_type()));
    TT_FATAL(tensor.buffer() != nullptr, "PolyNorm3Backward: {} must have an allocated device buffer", name);
    TT_FATAL(tensor.device() == input.device(), "PolyNorm3Backward: {} must be on the same device as input", name);
    TT_FATAL(
        tensor.layout() == tt::tt_metal::Layout::TILE,
        "PolyNorm3Backward: {} must use TILE layout. Layout: {}",
        name,
        enchantum::to_string(tensor.layout()));
    TT_FATAL(
        tensor.dtype() == expected_dtype,
        "PolyNorm3Backward: {} must use {}. Data type: {}",
        name,
        enchantum::to_string(expected_dtype),
        enchantum::to_string(tensor.dtype()));
    TT_FATAL(
        tensor.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
        "PolyNorm3Backward: {} must use INTERLEAVED memory layout. Memory layout: {}",
        name,
        enchantum::to_string(tensor.memory_config().memory_layout()));
    TT_FATAL(
        tensor.buffer()->buffer_type() == ttnn::BufferType::DRAM,
        "PolyNorm3Backward: {} must be in DRAM. Buffer type: {}",
        name,
        enchantum::to_string(tensor.buffer()->buffer_type()));

    const auto tile = tensor.tensor_spec().tile();
    const auto face_shape = tile.get_face_shape();
    TT_FATAL(
        tile.get_height() == tt::constants::TILE_HEIGHT && tile.get_width() == tt::constants::TILE_WIDTH &&
            face_shape[0] == tt::constants::FACE_HEIGHT && face_shape[1] == tt::constants::FACE_WIDTH &&
            tile.get_num_faces() == tt::constants::TILE_HW / tt::constants::FACE_HW &&
            !tile.get_transpose_within_face() && !tile.get_transpose_of_faces(),
        "PolyNorm3Backward: {} must use the canonical untransposed 32x32 tile with 16x16 faces",
        name);
}

tt::tt_metal::TensorSpec canonical_input_like_spec(const ttnn::Tensor& input) {
    return {
        input.logical_shape(),
        tt::tt_metal::TensorLayout(input.dtype(), tt::tt_metal::Layout::TILE, input.memory_config())};
}

tt::tt_metal::TensorSpec canonical_packed_partials_spec(const ttnn::Tensor& input) {
    const auto input_shape = input.logical_shape().to_array_4D();
    return {
        ttnn::Shape({input_shape[0], input_shape[1], input_shape[2], 128U}),
        tt::tt_metal::TensorLayout(tt::tt_metal::DataType::FLOAT32, tt::tt_metal::Layout::TILE, input.memory_config())};
}

}  // namespace

void PolyNorm3BackwardDeviceOperation::validate_on_program_cache_miss(
    const PolyNorm3BWAttributes&, const PolyNorm3BWTensorArgs& tensor_args) {
    const auto& input = tensor_args.input;
    validate_tensor(input, "input", ttnn::DataType::BFLOAT16, input);
    TT_FATAL(input.logical_shape().rank() == 4U, "PolyNorm3Backward: input must have rank 4");
    const auto input_shape = input.logical_shape().to_array_4D();
    TT_FATAL(
        input_shape[0] > 0U && input_shape[1] > 0U && input_shape[2] > 0U && input_shape[3] > 0U,
        "PolyNorm3Backward: all input dimensions must be positive. Shape: {}",
        input.logical_shape());
    TT_FATAL(
        input_shape[3] % tt::constants::TILE_WIDTH == 0U,
        "PolyNorm3Backward: input channels must be divisible by {}. Shape: {}",
        tt::constants::TILE_WIDTH,
        input.logical_shape());

    const auto expected_dL_dx_spec = canonical_input_like_spec(input);
    TT_FATAL(
        input.padded_shape() == expected_dL_dx_spec.padded_shape(),
        "PolyNorm3Backward: input padded shape {} must use canonical tile padding {}",
        input.padded_shape(),
        expected_dL_dx_spec.padded_shape());

    validate_tensor(tensor_args.dL_dout, "dL_dout", ttnn::DataType::BFLOAT16, input);
    TT_FATAL(
        tensor_args.dL_dout.tensor_spec() == input.tensor_spec(),
        "PolyNorm3Backward: dL_dout spec must exactly match input spec");
    validate_tensor(tensor_args.weight, "weight", ttnn::DataType::BFLOAT16, input);
    TT_FATAL(
        tensor_args.weight.logical_shape() == ttnn::Shape({1U, 1U, 1U, 3U}),
        "PolyNorm3Backward: weight must have shape [1, 1, 1, 3]. Shape: {}",
        tensor_args.weight.logical_shape());

    const auto expected_packed_partials_spec = canonical_packed_partials_spec(input);

    if (tensor_args.preallocated_dL_dx.has_value()) {
        const auto& preallocated_dL_dx = tensor_args.preallocated_dL_dx.value();
        validate_tensor(preallocated_dL_dx, "preallocated dL_dx", ttnn::DataType::BFLOAT16, input);
        TT_FATAL(
            preallocated_dL_dx.tensor_spec() == expected_dL_dx_spec,
            "PolyNorm3Backward: preallocated dL_dx spec must exactly match the canonical input-derived spec");
    }
    if (tensor_args.preallocated_packed_partials.has_value()) {
        const auto& packed_partials = tensor_args.preallocated_packed_partials.value();
        validate_tensor(packed_partials, "preallocated packed partials", ttnn::DataType::FLOAT32, input);
        TT_FATAL(
            packed_partials.tensor_spec() == expected_packed_partials_spec,
            "PolyNorm3Backward: preallocated packed partials spec must exactly match the canonical input-derived "
            "spec");
    }
}

PolyNorm3BWSpecReturn PolyNorm3BackwardDeviceOperation::compute_output_specs(
    const PolyNorm3BWAttributes&, const PolyNorm3BWTensorArgs& tensor_args) {
    PolyNorm3BWSpecReturn output_specs;
    output_specs.reserve(2U);

    if (tensor_args.preallocated_dL_dx.has_value()) {
        output_specs.push_back(tensor_args.preallocated_dL_dx->tensor_spec());
    } else {
        output_specs.emplace_back(
            tensor_args.input.logical_shape(),
            tt::tt_metal::TensorLayout(
                tensor_args.input.dtype(), tt::tt_metal::Layout::TILE, tensor_args.input.memory_config()));
    }

    if (tensor_args.preallocated_packed_partials.has_value()) {
        output_specs.push_back(tensor_args.preallocated_packed_partials->tensor_spec());
    } else {
        const auto input_shape = tensor_args.input.logical_shape().to_array_4D();
        output_specs.emplace_back(
            ttnn::Shape({input_shape[0], input_shape[1], input_shape[2], 128U}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::FLOAT32, tt::tt_metal::Layout::TILE, tensor_args.input.memory_config()));
    }
    return output_specs;
}

PolyNorm3BWTensorReturn PolyNorm3BackwardDeviceOperation::create_output_tensors(
    const PolyNorm3BWAttributes& op_attrs, const PolyNorm3BWTensorArgs& tensor_args) {
    PolyNorm3BWTensorReturn output_tensors;
    output_tensors.reserve(2U);
    auto specs = compute_output_specs(op_attrs, tensor_args);

    if (tensor_args.preallocated_dL_dx.has_value()) {
        output_tensors.push_back(tensor_args.preallocated_dL_dx.value());
    } else {
        output_tensors.push_back(ttnn::create_device_tensor(specs[0], tensor_args.input.device()));
    }

    if (tensor_args.preallocated_packed_partials.has_value()) {
        output_tensors.push_back(tensor_args.preallocated_packed_partials.value());
    } else {
        output_tensors.push_back(ttnn::create_device_tensor(specs[1], tensor_args.input.device()));
    }
    return output_tensors;
}

}  // namespace ttml::metal::ops::polynorm3_bw::device

namespace ttnn::prim {

ttml::metal::ops::polynorm3_bw::device::PolyNorm3BackwardDeviceOperation::tensor_return_value_t ttml_polynorm3_bw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& dL_dout_tensor,
    const ttnn::Tensor& weight_tensor,
    float epsilon,
    const std::optional<ttnn::Tensor>& preallocated_dL_dx,
    const std::optional<ttnn::Tensor>& preallocated_packed_partials) {
    using OperationType = ttml::metal::ops::polynorm3_bw::device::PolyNorm3BackwardDeviceOperation;

    const auto operation_attributes = OperationType::operation_attributes_t{
        .epsilon = epsilon,
    };
    const auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .dL_dout = dL_dout_tensor,
        .weight = weight_tensor,
        .preallocated_dL_dx = preallocated_dL_dx,
        .preallocated_packed_partials = preallocated_packed_partials,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
