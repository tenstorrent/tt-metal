// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "adamw_fused_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "adamw_fused_program_factory.hpp"

namespace ttml::metal::optimizers::adamw_fused::device {

AdamWFusedDeviceOperation::program_factory_t AdamWFusedDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return AdamWFusedProgramFactory{};
}

void AdamWFusedDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void AdamWFusedDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto check_tensor = [](const ttnn::Tensor& tensor,
                           const std::string& name,
                           const tt::tt_metal::Layout required_layout,
                           const tt::tt_metal::DataType required_dtype) {
        TT_FATAL(
            tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
            "AdamW optimizer requires '{}' to be on DEVICE. Got storage type: '{}'",
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
    const auto& exp_avg = tensor_args.exp_avg;
    const auto& exp_avg_sq = tensor_args.exp_avg_sq;
    check_tensor(param, "Parameter", tt::tt_metal::Layout::TILE, tt::tt_metal::DataType::BFLOAT16);
    check_tensor(grad, "Gradient", tt::tt_metal::Layout::TILE, tt::tt_metal::DataType::BFLOAT16);
    check_tensor(exp_avg, "Exponential Average Buffer", tt::tt_metal::Layout::TILE, tt::tt_metal::DataType::BFLOAT16);
    check_tensor(
        exp_avg_sq, "Exponential Average Squared Buffer", tt::tt_metal::Layout::TILE, tt::tt_metal::DataType::BFLOAT16);
}

AdamWFusedDeviceOperation::spec_return_value_t AdamWFusedDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tensor_args.param.tensor_spec();
}

AdamWFusedDeviceOperation::tensor_return_value_t AdamWFusedDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tensor_args.param;
}

ttsl::hash::hash_t AdamWFusedDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& param_tensor = tensor_args.param;
    const auto& param_logical_shape = param_tensor.logical_shape();
    auto program_factory = select_program_factory(args, tensor_args);
    auto hash = tt::tt_metal::operation::hash_operation<AdamWFusedDeviceOperation>(
        program_factory.index(), param_tensor.dtype(), param_logical_shape);

    return hash;
}

std::tuple<operation_attributes_t, tensor_args_t> AdamWFusedDeviceOperation::invoke(
    const ttnn::Tensor& param,
    const ttnn::Tensor& grad,
    const ttnn::Tensor& exp_avg,
    const ttnn::Tensor& exp_avg_sq,
    float lr,
    float beta1,
    float beta2,
    float beta1_pow,
    float beta2_pow,
    float epsilon,
    float weight_decay,
    uint32_t step) {
    return {
        operation_attributes_t{
            .lr = lr,
            .beta1 = beta1,
            .beta2 = beta2,
            .beta1_pow = beta1_pow,
            .beta2_pow = beta2_pow,
            .epsilon = epsilon,
            .weight_decay = weight_decay,
            .step = step,
        },
        tensor_args_t{
            .param = param,
            .grad = grad,
            .exp_avg = exp_avg,
            .exp_avg_sq = exp_avg_sq,
        }};
}

}  // namespace ttml::metal::optimizers::adamw_fused::device
