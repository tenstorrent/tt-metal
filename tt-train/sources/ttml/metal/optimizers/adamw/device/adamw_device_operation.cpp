// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "adamw_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "adamw_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::optimizers::adamw::device {

void AdamWDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto check_tensor = [](const ttnn::Tensor& tensor,
                           const std::string& name,
                           const tt::tt_metal::Layout required_layout,
                           const tt::tt_metal::DataType required_dtype) {
        TT_FATAL(
            tensor.storage_type() == ttnn::StorageType::DEVICE,
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
            tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
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

    const auto& beta1_pow = tensor_args.beta1_pow;
    const auto& beta2_pow = tensor_args.beta2_pow;
    TT_FATAL(
        beta1_pow.has_value() == beta2_pow.has_value(),
        "AdamW requires beta1_pow and beta2_pow to be supplied together, got beta1_pow={} beta2_pow={}",
        beta1_pow.has_value(),
        beta2_pow.has_value());

    if (beta1_pow.has_value()) {
        auto check_bias_tensor = [&](const ttnn::Tensor& tensor, const std::string& name) {
            TT_FATAL(
                tensor.dtype() == tt::tt_metal::DataType::FLOAT32,
                "Tensor '{}' must be FLOAT32, but got '{}'. The bias correction 1 - beta^t cancels badly in "
                "lower precision - bfloat16 rounds beta^t to 1.0 for small step counts, which would divide by zero.",
                name,
                enchantum::to_string(tensor.dtype()));
            check_tensor(tensor, name, tt::tt_metal::Layout::TILE, tt::tt_metal::DataType::FLOAT32);
            TT_FATAL(
                tensor.logical_volume() == 1U,
                "Tensor '{}' must hold exactly one element, got {}",
                name,
                tensor.logical_volume());
        };
        check_bias_tensor(beta1_pow.value(), "Beta1 Bias Correction");
        check_bias_tensor(beta2_pow.value(), "Beta2 Bias Correction");
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
    auto amsgrad = args.amsgrad;
    auto stochastic_rounding = args.stochastic_rounding;
    auto max_exp_avg_sq_initialized = tensor_args.max_exp_avg_sq.has_value();
    auto bias_correction_on_device = tensor_args.beta1_pow.has_value();
    auto hash = tt::tt_metal::operation::hash_operation<AdamWDeviceOperation>(
        amsgrad,
        stochastic_rounding,
        max_exp_avg_sq_initialized,
        bias_correction_on_device,
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
    ttml::metal::StochasticRounding stochastic_rounding,
    std::optional<uint32_t> stochastic_rounding_seed,
    const std::optional<ttnn::Tensor>& beta1_pow_tensor,
    const std::optional<ttnn::Tensor>& beta2_pow_tensor) {
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
        .stochastic_rounding_seed = stochastic_rounding_seed,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .param = param,
        .grad = grad,
        .exp_avg = exp_avg,
        .exp_avg_sq = exp_avg_sq,
        .max_exp_avg_sq = max_exp_avg_sq,
        .beta1_pow = beta1_pow_tensor,
        .beta2_pow = beta2_pow_tensor,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
