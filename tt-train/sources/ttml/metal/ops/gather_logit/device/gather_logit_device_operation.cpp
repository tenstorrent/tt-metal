// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_logit_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "gather_logit_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::gather_logit::device {

void GatherLogitDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto check_tensor = [](const ttnn::Tensor& tensor,
                           const std::string& name,
                           tt::tt_metal::Layout required_layout,
                           tt::tt_metal::DataType required_dtype) {
        TT_FATAL(
            tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
            "GatherLogit: '{}' must be on DEVICE, got '{}'",
            name,
            enchantum::to_string(tensor.storage_type()));
        TT_FATAL(tensor.buffer() != nullptr, "GatherLogit: '{}' buffer is null.", name);
        TT_FATAL(
            tensor.layout() == required_layout,
            "GatherLogit: '{}' must have layout '{}', got '{}'",
            name,
            enchantum::to_string(required_layout),
            enchantum::to_string(tensor.layout()));
        TT_FATAL(
            tensor.dtype() == required_dtype,
            "GatherLogit: '{}' must have dtype '{}', got '{}'",
            name,
            enchantum::to_string(required_dtype),
            enchantum::to_string(tensor.dtype()));
        TT_FATAL(
            tensor.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
            "GatherLogit: '{}' must use INTERLEAVED memory layout, got '{}'",
            name,
            enchantum::to_string(tensor.memory_config().memory_layout()));
    };

    check_tensor(tensor_args.logit, "logit", tt::tt_metal::Layout::TILE, tt::tt_metal::DataType::BFLOAT16);
    check_tensor(tensor_args.target, "target", tt::tt_metal::Layout::ROW_MAJOR, tt::tt_metal::DataType::UINT32);

    TT_FATAL(
        tensor_args.logit.logical_shape().rank() == 4U,
        "GatherLogit: logit must be rank 4, got rank {}",
        tensor_args.logit.logical_shape().rank());

    TT_FATAL(
        args.first_v < args.last_v,
        "GatherLogit: first_v ({}) must be less than last_v ({})",
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

GatherLogitDeviceOperation::spec_return_value_t GatherLogitDeviceOperation::compute_output_specs(
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

GatherLogitDeviceOperation::tensor_return_value_t GatherLogitDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.logit.device());
}

ttsl::hash::hash_t GatherLogitDeviceOperation::compute_program_hash(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    // first_v/last_v are runtime args and don't affect the compiled kernel binary,
    // so we only hash the tensor properties that determine program structure.
    return tt::tt_metal::operation::hash_operation<GatherLogitDeviceOperation>(
        tensor_args.logit.dtype(), tensor_args.logit.logical_shape());
}

}  // namespace ttml::metal::ops::gather_logit::device

namespace ttnn::prim {

ttml::metal::ops::gather_logit::device::GatherLogitDeviceOperation::tensor_return_value_t ttml_gather_logit(
    const ttnn::Tensor& logit,
    const ttnn::Tensor& target,
    uint32_t first_v,
    uint32_t last_v,
    const std::optional<ttnn::Tensor>& preallocated_output) {
    using OperationType = ttml::metal::ops::gather_logit::device::GatherLogitDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{first_v, last_v};
    auto tensor_args =
        OperationType::tensor_args_t{.logit = logit, .target = target, .preallocated_output = preallocated_output};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
