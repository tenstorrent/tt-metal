// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sgd_fused_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "sgd_fused_program_factory.hpp"

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
            tensor.device()->arch() == tt::ARCH::WORMHOLE_B0,
            "SGD optimizer is only supported on Wormhole. Device arch: {}. Tensor name: {}",
            enchantum::to_string(tensor.device()->arch()),
            name);

        TT_FATAL(
            tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
            "SGD optimizer requires '{}' to be on DEVICE. Got storage type: '{}'",
            name,
            enchantum::to_string(tensor.storage_type()));

        TT_FATAL(tensor.buffer() != nullptr, "Tensor '{}' must be allocated on device (buffer is null).", name);

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

    const auto& param_in = tensor_args.param_in;
    const auto& param_out = tensor_args.param_out;
    check_tensor(param_in, "Input", tt::tt_metal::Layout::TILE, tt::tt_metal::DataType::BFLOAT16);
    if (param_out.has_value()) {
        check_tensor(
            param_out.value(), "Preallocated Output", tt::tt_metal::Layout::TILE, tt::tt_metal::DataType::BFLOAT16);
    }
}

SGDFusedDeviceOperation::spec_return_value_t SGDFusedDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.param_out.has_value()) {
        return tensor_args.param_out->tensor_spec();
    }
    auto param_in_logical_shape = tensor_args.param_in.logical_shape();
    return ttnn::TensorSpec(
        ttnn::Shape(param_in_logical_shape),
        tt::tt_metal::TensorLayout(
            tensor_args.param_in.dtype(), tt::tt_metal::Layout::TILE, tensor_args.param_in.memory_config()));
}

SGDFusedDeviceOperation::tensor_return_value_t SGDFusedDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    tensor_return_value_t output_tensor;

    spec_return_value_t output_specs = compute_output_specs(args, tensor_args);

    if (tensor_args.param_out.has_value()) {
        output_tensor = tensor_args.param_out.value();
    } else {
        output_tensor = create_device_tensor(output_specs, tensor_args.param_in.device());
    }

    return output_tensor;
}

ttsl::hash::hash_t SGDFusedDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& param_in_tensor = tensor_args.param_in;
    const auto& param_in_logical_shape = param_in_tensor.logical_shape();
    auto program_factory = select_program_factory(args, tensor_args);
    auto hash = tt::tt_metal::operation::hash_operation<SGDFusedDeviceOperation>(
        args, program_factory.index(), param_in_tensor.dtype(), param_in_logical_shape);

    return hash;
}

std::tuple<operation_attributes_t, tensor_args_t> SGDFusedDeviceOperation::invoke(
    const ttnn::Tensor& param_in,
    const ttnn::Tensor& grad,
    float lr,
    float momentum,
    float dampening,
    const std::optional<ttnn::Tensor>& param_out,
    const std::optional<ttnn::Tensor>& momentum_in,
    const std::optional<ttnn::Tensor>& momentum_out) {
    return {
        operation_attributes_t{
            .lr = lr,
            .momentum = momentum,
            .dampening = dampening,
        },
        tensor_args_t{
            .param_in = param_in,
            .grad = grad,
            .param_out = param_out,
            .momentum_in = momentum_in,
            .momentum_out = momentum_out,
        }};
}

}  // namespace ttml::metal::optimizers::sgd_fused::device
