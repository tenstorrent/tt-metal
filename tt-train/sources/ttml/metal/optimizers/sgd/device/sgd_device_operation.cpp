// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sgd_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "metal/common/tensor_validation.hpp"
#include "sgd_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::optimizers::sgd::device {

void SGDDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& param = tensor_args.param;
    auto check_tensor = [&param](
                            const ttnn::Tensor& tensor,
                            const std::string& name,
                            const tt::tt_metal::Layout required_layout,
                            const tt::tt_metal::DataType required_dtype) {
        check_device_tensor(
            tensor,
            "SGD",
            name,
            {.dtypes = {required_dtype}, .layout = required_layout, .buffer_type = tt::tt_metal::BufferType::DRAM});

        // Logical shapes must match for element-for-element correspondence with the parameter;
        // padding alone cannot tell apart tensors that round up to the same tile extent.
        TT_FATAL(
            tensor.logical_shape() == param.logical_shape(),
            "Tensor '{}' must match the parameter's logical shape. Expected {}, got {}",
            name,
            param.logical_shape(),
            tensor.logical_shape());

        // Tile counts and reader/writer extents are derived solely from the parameter tensor, so any
        // smaller companion tensor would be read or written past its allocation.
        TT_FATAL(
            tensor.padded_shape() == param.padded_shape(),
            "Tensor '{}' must match the parameter's padded shape. Expected {}, got {}",
            name,
            param.padded_shape(),
            tensor.padded_shape());
    };

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

SGDDeviceOperation::spec_return_value_t SGDDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tensor_args.param.tensor_spec();
}

SGDDeviceOperation::tensor_return_value_t SGDDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tensor_args.param;
}

ttsl::hash::hash_t SGDDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& param_tensor = tensor_args.param;
    const auto& param_logical_shape = param_tensor.logical_shape();
    auto nesterov = args.nesterov;
    auto momentum_initialized = tensor_args.momentum_buffer.has_value();
    auto hash = tt::tt_metal::operation::hash_operation<SGDDeviceOperation>(
        nesterov, momentum_initialized, param_tensor.dtype(), param_logical_shape);

    return hash;
}

}  // namespace ttml::metal::optimizers::sgd::device

namespace ttnn::prim {

ttml::metal::optimizers::sgd::device::SGDDeviceOperation::tensor_return_value_t sgd(
    const ttnn::Tensor& param,
    const ttnn::Tensor& grad,
    float lr,
    float momentum,
    float dampening,
    float weight_decay,
    bool nesterov,
    const std::optional<ttnn::Tensor>& momentum_buffer) {
    using OperationType = ttml::metal::optimizers::sgd::device::SGDDeviceOperation;

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
