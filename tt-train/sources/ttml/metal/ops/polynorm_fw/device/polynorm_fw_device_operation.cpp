// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "polynorm_fw_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::polynorm3_fw::device {

void PolyNorm3ForwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto check_tensor = [](const ttnn::Tensor& tensor, const std::string& name) {
        TT_FATAL(
            tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
            "PolyNorm3Forward operation requires {} to be on Device. Input storage type: {}",
            name,
            enchantum::to_string(tensor.storage_type()));
        TT_FATAL(
            tensor.buffer() != nullptr,
            "Operands to PolyNorm3Forward need to be allocated in buffers on the device. Buffer is null. Tensor name "
            "{}",
            name);
        TT_FATAL(
            tensor.layout() == tt::tt_metal::Layout::TILE,
            "PolyNorm3Forward operation requires tensor to be in Tile layout. {} tensor layout: {}",
            name,
            enchantum::to_string(tensor.layout()));
        TT_FATAL(
            tensor.dtype() == tt::tt_metal::DataType::BFLOAT16,
            "PolyNorm3Forward operation requires tensor to be of BFLOAT16 data type. {} tensor data type: {}",
            name,
            enchantum::to_string(tensor.dtype()));
        TT_FATAL(
            tensor.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
            "PolyNorm3Forward operation requires Interleaved memory layout. {} memory layout: `{}`",
            name,
            enchantum::to_string(tensor.memory_config().memory_layout()));
    };

    check_tensor(tensor_args.input, "Input");
    check_tensor(tensor_args.weight, "weight");
    check_tensor(tensor_args.bias, "bias");
    if (tensor_args.preallocated_output.has_value()) {
        check_tensor(tensor_args.preallocated_output.value(), "Preallocated output");
    }
}

spec_return_value_t PolyNorm3ForwardDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return {tensor_args.preallocated_output->tensor_spec()};
    }
    return {ttnn::TensorSpec(
        tensor_args.input.logical_shape(),
        tt::tt_metal::TensorLayout(
            tensor_args.input.dtype(), tt::tt_metal::Layout::TILE, tensor_args.input.memory_config()))};
}

tensor_return_value_t PolyNorm3ForwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& op_attrs, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    auto specs = compute_output_specs(op_attrs, tensor_args);
    return create_device_tensor(specs[0], tensor_args.input.device());
}

ttsl::hash::hash_t PolyNorm3ForwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    return tt::tt_metal::operation::hash_operation<PolyNorm3ForwardDeviceOperation>(
        args, input.dtype(), input.logical_shape());
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
