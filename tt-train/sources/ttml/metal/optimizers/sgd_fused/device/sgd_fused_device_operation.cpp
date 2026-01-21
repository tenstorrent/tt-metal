// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sgd_fused_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "sgd_fused_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::optimizers::sgd_fused::device {

SGDFusedDeviceOperation::program_factory_t SGDFusedDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return SGDFusedProgramFactory{};
}

void SGDFusedDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void SGDFusedDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto check_tensor = [](const ttnn::Tensor& tensor,
                           const std::string& name,
                           const tt::tt_metal::Layout required_layout,
                           const tt::tt_metal::DataType required_dtype) {
        TT_FATAL(
            tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
            "SGD optimizer requires '{}' to be on DEVICE. Got storage type: '{}'",
            name,
            enchantum::to_string(tensor.storage_type()));

        TT_FATAL(tensor.buffer() != nullptr, "Tensor '{}' must be allocated on device (buffer is null).", name);

        TT_FATAL(
            tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "Tensor '{}' must be in DRAM. Got buffer type: '{}'",
            name,
            enchantum::to_string(tensor.buffer()->buffer_type()));

        TT_FATAL(
            tensor.layout() == required_layout,
            "Tensor '{}' must have layout '{}', but got '{}'",
            name,
            enchantum::to_string(required_layout),
            enchantum::to_string(tensor.layout()));

        TT_FATAL(
            tensor.dtype() == required_dtype,
            "Tensor '{}' must have data type '{}', but got '{}'",
            name,
            enchantum::to_string(required_dtype),
            enchantum::to_string(tensor.dtype()));

        TT_FATAL(
            tensor.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
            "Tensor '{}' must use INTERLEAVED memory layout, but got '{}'",
            name,
            enchantum::to_string(tensor.memory_config().memory_layout()));
    };

    const auto& param = tensor_args.param;
    const auto& grad = tensor_args.grad;
    const auto& momentum_buffer = tensor_args.momentum_buffer;
    check_tensor(param, "Parameter", tt::tt_metal::Layout::TILE, tt::tt_metal::DataType::BFLOAT16);
    check_tensor(grad, "Gradient", tt::tt_metal::Layout::TILE, tt::tt_metal::DataType::BFLOAT16);
    if (momentum_buffer.has_value()) {
        check_tensor(
            momentum_buffer.value(), "Momentum Buffer", tt::tt_metal::Layout::TILE, tt::tt_metal::DataType::BFLOAT16);
    }

    const auto momentum = args.momentum;
    const auto use_momentum = (momentum > 0.0F);
    if (use_momentum) {
        TT_FATAL(
            momentum_buffer.has_value(),
            "Momentum buffer must be provided when using momentum. Got momentum value: {}. Please set momentum to "
            "zero or pass momentum buffer.",
            momentum);
    }
}

SGDFusedDeviceOperation::spec_return_value_t SGDFusedDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tensor_args.param.tensor_spec();
}

SGDFusedDeviceOperation::tensor_return_value_t SGDFusedDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tensor_args.param;
}

ttsl::hash::hash_t SGDFusedDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& param_tensor = tensor_args.param;
    const auto& param_logical_shape = param_tensor.logical_shape();
    auto program_factory = select_program_factory(args, tensor_args);
    auto nesterov = args.nesterov;
    auto momentum_initialized = tensor_args.momentum_buffer.has_value();
    auto hash = tt::tt_metal::operation::hash_operation<SGDFusedDeviceOperation>(
        nesterov, momentum_initialized, program_factory.index(), param_tensor.dtype(), param_logical_shape);

    return hash;
}

}  // namespace ttml::metal::optimizers::sgd_fused::device

namespace ttnn::prim {

ttml::metal::optimizers::sgd_fused::device::SGDFusedDeviceOperation::tensor_return_value_t ttml_sgd_fused(
    const ttnn::Tensor& param,
    const ttnn::Tensor& grad,
    float lr,
    float momentum,
    float dampening,
    float weight_decay,
    bool nesterov,
    const std::optional<ttnn::Tensor>& momentum_buffer) {
    using OperationType = ttml::metal::optimizers::sgd_fused::device::SGDFusedDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .lr = lr,
        .momentum = momentum,
        .dampening = dampening,
        .weight_decay = weight_decay,
        .nesterov = nesterov,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .param = param,
        .grad = grad,
        .momentum_buffer = momentum_buffer,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
