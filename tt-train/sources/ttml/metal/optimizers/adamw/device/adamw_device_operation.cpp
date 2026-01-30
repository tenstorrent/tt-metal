// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "adamw_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "adamw_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::optimizers::adamw::device {

AdamWDeviceOperation::program_factory_t AdamWDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return AdamWProgramFactory{};
}

void AdamWDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void AdamWDeviceOperation::validate_on_program_cache_miss(
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
    const auto& max_exp_avg_sq = tensor_args.max_exp_avg_sq;

    // Determine the precision mode based on param dtype
    const auto param_dtype = param.dtype();
    const bool is_half_precision = (param_dtype == tt::tt_metal::DataType::BFLOAT16);

    // Validate param dtype is either bf16 or fp32
    TT_FATAL(
        param_dtype == tt::tt_metal::DataType::BFLOAT16 || param_dtype == tt::tt_metal::DataType::FLOAT32,
        "Parameter tensor must be BFLOAT16 or FLOAT32, but got '{}'",
        enchantum::to_string(param_dtype));

    // Stochastic rounding is only valid for half precision (bf16) mode
    TT_FATAL(
        args.stochastic_rounding == StochasticRounding::Disabled || is_half_precision,
        "Stochastic rounding is only supported with BFLOAT16 parameters. "
        "Got stochastic_rounding=Enabled with parameter dtype '{}'",
        enchantum::to_string(param_dtype));

    // Validate all tensors
    check_tensor(param, "Parameter", tt::tt_metal::Layout::TILE, param_dtype);
    // Gradient is always bf16
    check_tensor(grad, "Gradient", tt::tt_metal::Layout::TILE, tt::tt_metal::DataType::BFLOAT16);
    // Optimizer states must match param dtype
    check_tensor(exp_avg, "Exponential Average Buffer", tt::tt_metal::Layout::TILE, param_dtype);
    check_tensor(exp_avg_sq, "Exponential Average Squared Buffer", tt::tt_metal::Layout::TILE, param_dtype);

    if (max_exp_avg_sq.has_value()) {
        check_tensor(
            max_exp_avg_sq.value(), "Max Exponential Average Squared Buffer", tt::tt_metal::Layout::TILE, param_dtype);
    }
}

AdamWDeviceOperation::spec_return_value_t AdamWDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tensor_args.param.tensor_spec();
}

AdamWDeviceOperation::tensor_return_value_t AdamWDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tensor_args.param;
}

ttsl::hash::hash_t AdamWDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& param_tensor = tensor_args.param;
    const auto& param_logical_shape = param_tensor.logical_shape();
    auto program_factory = select_program_factory(args, tensor_args);
    auto amsgrad = args.amsgrad;
    auto stochastic_rounding = args.stochastic_rounding;
    auto max_exp_avg_sq_initialized = tensor_args.max_exp_avg_sq.has_value();
    auto hash = tt::tt_metal::operation::hash_operation<AdamWDeviceOperation>(
        amsgrad,
        stochastic_rounding,
        max_exp_avg_sq_initialized,
        program_factory.index(),
        param_tensor.dtype(),
        param_logical_shape);

    return hash;
}

}  // namespace ttml::metal::optimizers::adamw::device

namespace ttnn::prim {

ttml::metal::optimizers::adamw::device::AdamWDeviceOperation::tensor_return_value_t adamw(
    const ttnn::Tensor& param,
    const ttnn::Tensor& grad,
    const ttnn::Tensor& exp_avg,
    const ttnn::Tensor& exp_avg_sq,
    const std::optional<ttnn::Tensor>& max_exp_avg_sq,
    float lr,
    float beta1,
    float beta2,
    float beta1_pow,
    float beta2_pow,
    float epsilon,
    float weight_decay,
    bool amsgrad,
    ttml::metal::StochasticRounding stochastic_rounding) {
    using OperationType = ttml::metal::optimizers::adamw::device::AdamWDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .lr = lr,
        .beta1 = beta1,
        .beta2 = beta2,
        .beta1_pow = beta1_pow,
        .beta2_pow = beta2_pow,
        .epsilon = epsilon,
        .weight_decay = weight_decay,
        .amsgrad = amsgrad,
        .stochastic_rounding = stochastic_rounding,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .param = param,
        .grad = grad,
        .exp_avg = exp_avg,
        .exp_avg_sq = exp_avg_sq,
        .max_exp_avg_sq = max_exp_avg_sq,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
