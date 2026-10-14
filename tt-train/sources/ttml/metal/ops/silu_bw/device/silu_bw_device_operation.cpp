// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "silu_bw_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "silu_bw_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::silu_bw::device {

void SiLUBackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto check_tensor = [](const ttnn::Tensor& tensor, const std::string& name) {
        TT_FATAL(
            tensor.storage_type() == ttnn::StorageType::DEVICE,
            "SiLUBackward operation requires {} to be on Device. Input storage type: {}",
            name,
            enchantum::to_string(tensor.storage_type()));

        TT_FATAL(
            tensor.buffer() != nullptr,
            "Operands to SiLUBackward need to be allocated in buffers on the device. Buffer is null. Tensor name {}",
            name);

        TT_FATAL(
            tensor.layout() == tt::tt_metal::Layout::TILE,
            "SiLUBackward operation requires tensor to be in Tile layout. {} tensor layout: {}",
            name,
            enchantum::to_string(tensor.layout()));

        TT_FATAL(
            tensor.dtype() == tt::tt_metal::DataType::BFLOAT16,
            "SiLUBackward operation requires tensor to be of BFLOAT16 data type. {} tensor data type: {}",
            name,
            enchantum::to_string(tensor.dtype()));

        TT_FATAL(
            tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
            "SiLUBackward operation requires Interleaved memory layout. {} "
            "memory layout: `{}`",
            name,
            enchantum::to_string(tensor.memory_config().memory_layout()));
    };

    const auto& input_tensor = tensor_args.input;
    const auto& dL_dout_tensor = tensor_args.dL_dout;
    const auto& preallocated_da_tensor = tensor_args.preallocated_da;

    check_tensor(input_tensor, "Input");
    check_tensor(dL_dout_tensor, "dL_dout");

    const auto& expected_logical_shape = input_tensor.logical_shape();
    const auto& expected_padded_shape = input_tensor.padded_shape();
    TT_FATAL(
        dL_dout_tensor.logical_shape() == expected_logical_shape,
        "SiLUBackward: dL_dout logical shape {} does not match input logical shape {}",
        dL_dout_tensor.logical_shape(),
        expected_logical_shape);
    TT_FATAL(
        dL_dout_tensor.padded_shape() == expected_padded_shape,
        "SiLUBackward: dL_dout padded shape {} does not match input padded shape {}",
        dL_dout_tensor.padded_shape(),
        expected_padded_shape);

    if (preallocated_da_tensor.has_value()) {
        check_tensor(preallocated_da_tensor.value(), "Preallocated dL_da");
        TT_FATAL(
            preallocated_da_tensor->logical_shape() == expected_logical_shape,
            "SiLUBackward: preallocated dL_da logical shape {} does not match input logical shape {}",
            preallocated_da_tensor->logical_shape(),
            expected_logical_shape);
        TT_FATAL(
            preallocated_da_tensor->padded_shape() == expected_padded_shape,
            "SiLUBackward: preallocated dL_da padded shape {} does not match input padded shape {}",
            preallocated_da_tensor->padded_shape(),
            expected_padded_shape);
    }
}

spec_return_value_t SiLUBackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    spec_return_value_t output_specs;
    output_specs.reserve(2U);

    if (tensor_args.preallocated_da.has_value()) {
        output_specs.push_back(tensor_args.preallocated_da->tensor_spec());
    } else {
        output_specs.push_back(tensor_args.input.tensor_spec());
    }

    return output_specs;
}

tensor_return_value_t SiLUBackwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    spec_return_value_t output_specs = compute_output_specs(args, tensor_args);

    if (tensor_args.preallocated_da.has_value()) {
        return tensor_args.preallocated_da.value();
    } else {
        return ttnn::create_device_tensor(output_specs[0], tensor_args.input.device());
    }
}

ttsl::hash::hash_t SiLUBackwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& input_logical_shape = input_tensor.logical_shape();
    const auto& input_padded_shape = input_tensor.padded_shape();
    tt::tt_metal::operation::Hash hash = tt::tt_metal::operation::hash_operation<SiLUBackwardDeviceOperation>(
        args, input_tensor.dtype(), input_logical_shape, input_padded_shape);

    return hash;
}

}  // namespace ttml::metal::ops::silu_bw::device

namespace ttnn::prim {

ttml::metal::ops::silu_bw::device::SiLUBackwardDeviceOperation::tensor_return_value_t ttml_silu_bw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& dL_dout_tensor,
    const std::optional<ttnn::Tensor>& preallocated_da) {
    using OperationType = ttml::metal::ops::silu_bw::device::SiLUBackwardDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{};
    auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .dL_dout = dL_dout_tensor,
        .preallocated_da = preallocated_da,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
