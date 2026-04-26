// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "select_target_logit_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "select_target_logit_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::select_target_logit::device {

void SelectTargetLogitDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto check_tensor = [](const ttnn::Tensor& tensor,
                           const std::string& name,
                           tt::tt_metal::Layout required_layout,
                           tt::tt_metal::DataType required_dtype) {
        TT_FATAL(
            tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
            "SelectTargetLogit: '{}' must be on DEVICE, got '{}'",
            name,
            enchantum::to_string(tensor.storage_type()));
        TT_FATAL(tensor.buffer() != nullptr, "SelectTargetLogit: '{}' buffer is null.", name);
        TT_FATAL(
            tensor.layout() == required_layout,
            "SelectTargetLogit: '{}' must have layout '{}', got '{}'",
            name,
            enchantum::to_string(required_layout),
            enchantum::to_string(tensor.layout()));
        TT_FATAL(
            tensor.dtype() == required_dtype,
            "SelectTargetLogit: '{}' must have dtype '{}', got '{}'",
            name,
            enchantum::to_string(required_dtype),
            enchantum::to_string(tensor.dtype()));
        TT_FATAL(
            tensor.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
            "SelectTargetLogit: '{}' must use INTERLEAVED memory layout, got '{}'",
            name,
            enchantum::to_string(tensor.memory_config().memory_layout()));
    };

    check_tensor(tensor_args.logit, "logit", tt::tt_metal::Layout::TILE, tt::tt_metal::DataType::BFLOAT16);
    check_tensor(tensor_args.target, "target", tt::tt_metal::Layout::ROW_MAJOR, tt::tt_metal::DataType::UINT32);

    TT_FATAL(
        tensor_args.logit.logical_shape().rank() == 4U,
        "SelectTargetLogit: logit must be rank 4, got rank {}",
        tensor_args.logit.logical_shape().rank());

    TT_FATAL(
        args.first_v < args.last_v,
        "SelectTargetLogit: first_v ({}) must be less than last_v ({})",
        args.first_v,
        args.last_v);

    if (tensor_args.preallocated_output.has_value()) {
        const auto& out = tensor_args.preallocated_output.value();
        TT_FATAL(
            out.storage_type() == tt::tt_metal::StorageType::DEVICE,
            "SelectTargetLogit: 'preallocated_output' must be on DEVICE, got '{}'",
            enchantum::to_string(out.storage_type()));
        TT_FATAL(out.buffer() != nullptr, "SelectTargetLogit: 'preallocated_output' buffer is null.");
        TT_FATAL(
            out.layout() == tt::tt_metal::Layout::TILE,
            "SelectTargetLogit: 'preallocated_output' must have layout 'TILE', got '{}'",
            enchantum::to_string(out.layout()));
        TT_FATAL(
            out.dtype() == tt::tt_metal::DataType::BFLOAT16,
            "SelectTargetLogit: 'preallocated_output' must be BFLOAT16, got '{}'",
            enchantum::to_string(out.dtype()));
        TT_FATAL(
            out.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
            "SelectTargetLogit: 'preallocated_output' must use INTERLEAVED memory layout, got '{}'",
            enchantum::to_string(out.memory_config().memory_layout()));
    }
}

SelectTargetLogitDeviceOperation::spec_return_value_t SelectTargetLogitDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }
    auto shape = tensor_args.logit.logical_shape();
    shape[-1] = 1U;
    return ttnn::TensorSpec(
        ttnn::Shape(shape),
        tt::tt_metal::TensorLayout(
            tensor_args.logit.dtype(), tt::tt_metal::Layout::TILE, tensor_args.logit.memory_config()));
}

SelectTargetLogitDeviceOperation::tensor_return_value_t SelectTargetLogitDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.logit.device());
}

ttsl::hash::hash_t SelectTargetLogitDeviceOperation::compute_program_hash(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    // first_v/last_v are runtime args and don't affect the compiled kernel binary.
    return tt::tt_metal::operation::hash_operation<SelectTargetLogitDeviceOperation>(
        tensor_args.logit.dtype(), tensor_args.logit.logical_shape());
}

}  // namespace ttml::metal::ops::select_target_logit::device

namespace ttnn::prim {

ttml::metal::ops::select_target_logit::device::SelectTargetLogitDeviceOperation::tensor_return_value_t
ttml_select_target_logit(
    const ttnn::Tensor& logit,
    const ttnn::Tensor& target,
    uint32_t first_v,
    uint32_t last_v,
    const std::optional<ttnn::Tensor>& preallocated_output) {
    using OperationType = ttml::metal::ops::select_target_logit::device::SelectTargetLogitDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{first_v, last_v};
    auto tensor_args =
        OperationType::tensor_args_t{.logit = logit, .target = target, .preallocated_output = preallocated_output};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
