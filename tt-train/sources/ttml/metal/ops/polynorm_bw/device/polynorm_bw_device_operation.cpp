// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "polynorm_bw_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::polynorm3_bw::device {

void PolyNorm3BackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    auto check_tensor = [](const ttnn::Tensor& tensor, const std::string& name) {
        TT_FATAL(
            tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
            "PolyNormBackward operation requires {} to be on Device. Input storage type: {}",
            name,
            enchantum::to_string(tensor.storage_type()));
        TT_FATAL(
            tensor.buffer() != nullptr,
            "Operands to PolyNormBackward need to be allocated in buffers on the device. Buffer is null. Tensor name "
            "{}",
            name);
        TT_FATAL(
            tensor.layout() == tt::tt_metal::Layout::TILE,
            "PolyNormBackward operation requires tensor to be in Tile layout. {} tensor layout: {}",
            name,
            enchantum::to_string(tensor.layout()));
        TT_FATAL(
            tensor.dtype() == tt::tt_metal::DataType::BFLOAT16,
            "PolyNormBackward operation requires tensor to be of BFLOAT16 data type. {} tensor data type: {}",
            name,
            enchantum::to_string(tensor.dtype()));
        TT_FATAL(
            tensor.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
            "PolyNormBackward operation requires Interleaved memory layout. {} memory layout: `{}`",
            name,
            enchantum::to_string(tensor.memory_config().memory_layout()));
    };

    check_tensor(tensor_args.input, "Input");
    check_tensor(tensor_args.dL_dout, "dL_dout");
    check_tensor(tensor_args.weight, "Weight");
    if (tensor_args.preallocated_dL_dx.has_value()) {
        check_tensor(tensor_args.preallocated_dL_dx.value(), "Preallocated dL_dx");
    }
    if (tensor_args.preallocated_packed_partials.has_value()) {
        const auto& packed_partials = tensor_args.preallocated_packed_partials.value();
        TT_FATAL(
            packed_partials.layout() == tt::tt_metal::Layout::TILE,
            "Preallocated packed partials must be tile layout.");
        TT_FATAL(
            packed_partials.dtype() == tt::tt_metal::DataType::FLOAT32,
            "Preallocated packed partials must be FLOAT32.");
    }
}

spec_return_value_t PolyNorm3BackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    spec_return_value_t output_specs;
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

tensor_return_value_t PolyNorm3BackwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& op_attrs, const tensor_args_t& tensor_args) {
    tensor_return_value_t output_tensors;
    output_tensors.reserve(2U);
    auto specs = compute_output_specs(op_attrs, tensor_args);

    if (tensor_args.preallocated_dL_dx.has_value()) {
        output_tensors.push_back(tensor_args.preallocated_dL_dx.value());
    } else {
        output_tensors.push_back(create_device_tensor(specs[0], tensor_args.input.device()));
    }

    if (tensor_args.preallocated_packed_partials.has_value()) {
        output_tensors.push_back(tensor_args.preallocated_packed_partials.value());
    } else {
        output_tensors.push_back(create_device_tensor(specs[1], tensor_args.input.device()));
    }
    return output_tensors;
}

ttsl::hash::hash_t PolyNorm3BackwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    return tt::tt_metal::operation::hash_operation<PolyNorm3BackwardDeviceOperation>(
        args.epsilon, input.dtype(), input.logical_shape());
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
