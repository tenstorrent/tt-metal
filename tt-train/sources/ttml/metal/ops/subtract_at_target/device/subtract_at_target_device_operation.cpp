// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "subtract_at_target_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "subtract_at_target_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::subtract_at_target::device {

void SubtractAtTargetDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto check_tensor = [](const ttnn::Tensor& tensor,
                           const std::string& name,
                           tt::tt_metal::Layout required_layout,
                           tt::tt_metal::DataType required_dtype) {
        TT_FATAL(
            tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
            "SubtractAtTarget: '{}' must be on DEVICE, got '{}'",
            name,
            enchantum::to_string(tensor.storage_type()));
        TT_FATAL(tensor.buffer() != nullptr, "SubtractAtTarget: '{}' buffer is null.", name);
        TT_FATAL(
            tensor.layout() == required_layout,
            "SubtractAtTarget: '{}' must have layout '{}', got '{}'",
            name,
            enchantum::to_string(required_layout),
            enchantum::to_string(tensor.layout()));
        TT_FATAL(
            tensor.dtype() == required_dtype,
            "SubtractAtTarget: '{}' must have dtype '{}', got '{}'",
            name,
            enchantum::to_string(required_dtype),
            enchantum::to_string(tensor.dtype()));
        TT_FATAL(
            tensor.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
            "SubtractAtTarget: '{}' must use INTERLEAVED memory layout, got '{}'",
            name,
            enchantum::to_string(tensor.memory_config().memory_layout()));
    };

    check_tensor(tensor_args.input, "input", tt::tt_metal::Layout::TILE, tt::tt_metal::DataType::BFLOAT16);
    check_tensor(tensor_args.target, "target", tt::tt_metal::Layout::ROW_MAJOR, tt::tt_metal::DataType::UINT32);

    TT_FATAL(
        tensor_args.input.logical_shape().rank() == 4U,
        "SubtractAtTarget: input must be rank 4, got rank {}",
        tensor_args.input.logical_shape().rank());

    TT_FATAL(
        args.first_v < args.last_v,
        "SubtractAtTarget: first_v ({}) must be less than last_v ({})",
        args.first_v,
        args.last_v);

    if (tensor_args.preallocated_output.has_value()) {
        check_tensor(
            tensor_args.preallocated_output.value(),
            "preallocated_output",
            tt::tt_metal::Layout::TILE,
            tt::tt_metal::DataType::BFLOAT16);
    }
}

SubtractAtTargetDeviceOperation::spec_return_value_t SubtractAtTargetDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }
    return tensor_args.input.tensor_spec();
}

SubtractAtTargetDeviceOperation::tensor_return_value_t SubtractAtTargetDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

ttsl::hash::hash_t SubtractAtTargetDeviceOperation::compute_program_hash(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<SubtractAtTargetDeviceOperation>(
        tensor_args.input.dtype(), tensor_args.input.logical_shape());
}

}  // namespace ttml::metal::ops::subtract_at_target::device

namespace ttnn::prim {

ttml::metal::ops::subtract_at_target::device::SubtractAtTargetDeviceOperation::tensor_return_value_t
ttml_subtract_at_target(
    const ttnn::Tensor& input,
    const ttnn::Tensor& target,
    uint32_t first_v,
    uint32_t last_v,
    const std::optional<ttnn::Tensor>& preallocated_output,
    float subtract_value) {
    using OperationType = ttml::metal::ops::subtract_at_target::device::SubtractAtTargetDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{first_v, last_v, subtract_value};
    auto tensor_args =
        OperationType::tensor_args_t{.input = input, .target = target, .preallocated_output = preallocated_output};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
