// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "polynorm_bw_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::polynorm_bw::device {

void PolyNormBackwardDeviceOperation::validate_on_program_cache_miss(
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
    if (tensor_args.preallocated_dL_dx.has_value()) {
        check_tensor(tensor_args.preallocated_dL_dx.value(), "Preallocated dL_dx");
    }
}

spec_return_value_t PolyNormBackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_dL_dx.has_value()) {
        return {tensor_args.preallocated_dL_dx->tensor_spec()};
    }
    return {ttnn::TensorSpec(
        tensor_args.input.logical_shape(),
        tt::tt_metal::TensorLayout(
            tensor_args.input.dtype(), tt::tt_metal::Layout::TILE, tensor_args.input.memory_config()))};
}

tensor_return_value_t PolyNormBackwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& op_attrs, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_dL_dx.has_value()) {
        return tensor_args.preallocated_dL_dx.value();
    }
    auto specs = compute_output_specs(op_attrs, tensor_args);
    return create_device_tensor(specs[0], tensor_args.input.device());
}

ttsl::hash::hash_t PolyNormBackwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    return tt::tt_metal::operation::hash_operation<PolyNormBackwardDeviceOperation>(
        args, input.dtype(), input.logical_shape());
}

}  // namespace ttml::metal::ops::polynorm_bw::device

namespace ttnn::prim {

ttml::metal::ops::polynorm_bw::device::PolyNormBackwardDeviceOperation::tensor_return_value_t ttml_polynorm_bw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& dL_dout_tensor,
    float w0,
    float w1,
    float w2,
    float epsilon,
    const std::optional<ttnn::Tensor>& preallocated_dL_dx) {
    using OperationType = ttml::metal::ops::polynorm_bw::device::PolyNormBackwardDeviceOperation;

    const auto operation_attributes = OperationType::operation_attributes_t{
        .epsilon = epsilon,
        .w0 = w0,
        .w1 = w1,
        .w2 = w2,
    };
    const auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .dL_dout = dL_dout_tensor,
        .preallocated_dL_dx = preallocated_dL_dx,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
