// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_fw_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "layernorm_fw_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::layernorm_fw::device {

LayerNormForwardDeviceOperation::program_factory_t LayerNormForwardDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return LayerNormForwardProgramFactory{};
}

void LayerNormForwardDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void LayerNormForwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto check_tensor = [](const ttnn::Tensor& tensor, const std::string& name) {
        TT_FATAL(
            tensor.storage_type() == tt::tt_metal::StorageType::DEVICE,
            "Tensor's '{}' storage type must be {}. Got storage type: {}",
            name,
            enchantum::to_string(tt::tt_metal::StorageType::DEVICE),
            enchantum::to_string(tensor.storage_type()));

        TT_FATAL(tensor.buffer() != nullptr, "Tensor '{}' must be allocated on device (buffer is null).", name);

        TT_FATAL(
            tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "Tensor '{}' buffer must be in DRAM. Buffer of type {}",
            name,
            enchantum::to_string(tensor.buffer()->buffer_type()));

        TT_FATAL(
            tensor.layout() == tt::tt_metal::Layout::TILE,
            "Tensor '{}' must be in Tile layout. Got layout: {}",
            name,
            enchantum::to_string(tensor.layout()));

        TT_FATAL(
            tensor.dtype() == tt::tt_metal::DataType::BFLOAT16,
            "Tensor '{}' must be of BFLOAT16 data type. Got data type: {}",
            name,
            enchantum::to_string(tensor.dtype()));

        TT_FATAL(
            tensor.memory_config().memory_layout() == ttnn::TensorMemoryLayout::INTERLEAVED,
            "Tensor '{}' must use Interleaved memory layout. Got memory layout: {}",
            name,
            enchantum::to_string(tensor.memory_config().memory_layout()));
    };

    const auto& input_tensor = tensor_args.input;
    const auto& gamma_tensor = tensor_args.gamma;
    const auto& beta_tensor = tensor_args.beta;
    const auto& preallocated_output_tensor = tensor_args.preallocated_output;
    const auto& preallocated_mean_tensor = tensor_args.preallocated_mean;
    const auto& preallocated_rstd_tensor = tensor_args.preallocated_rstd;

    check_tensor(input_tensor, "Input");
    check_tensor(gamma_tensor, "Gamma");
    check_tensor(beta_tensor, "Beta");
    if (preallocated_output_tensor.has_value()) {
        check_tensor(preallocated_output_tensor.value(), "Preallocated output");
    }
    if (preallocated_mean_tensor.has_value()) {
        check_tensor(preallocated_mean_tensor.value(), "Preallocated mean");
    }
    if (preallocated_rstd_tensor.has_value()) {
        check_tensor(preallocated_rstd_tensor.value(), "Preallocated rstd");
    }
}

spec_return_value_t LayerNormForwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    spec_return_value_t output_specs;
    output_specs.reserve(3U);

    // output - same shape as input
    if (tensor_args.preallocated_output.has_value()) {
        output_specs.push_back(tensor_args.preallocated_output->tensor_spec());
    } else {
        auto input_shape = tensor_args.input.logical_shape();
        output_specs.emplace_back(
            input_shape,
            tt::tt_metal::TensorLayout(
                tensor_args.input.dtype(), tt::tt_metal::Layout::TILE, tensor_args.input.memory_config()));
    }

    // mean - shape is [B, 1, S, 1]
    if (args.return_mean_rstd) {
        if (tensor_args.preallocated_mean.has_value()) {
            output_specs.push_back(tensor_args.preallocated_mean->tensor_spec());
        } else {
            auto mean_shape = tensor_args.input.logical_shape();
            mean_shape[-1] = 1U;
            output_specs.emplace_back(
                mean_shape,
                tt::tt_metal::TensorLayout(
                    tensor_args.input.dtype(), tt::tt_metal::Layout::TILE, tensor_args.input.memory_config()));
        }

        // rstd - same shape as mean [B, H, S, 1]
        if (tensor_args.preallocated_rstd.has_value()) {
            output_specs.push_back(tensor_args.preallocated_rstd->tensor_spec());
        } else {
            auto rstd_shape = tensor_args.input.logical_shape();
            rstd_shape[-1] = 1U;
            output_specs.emplace_back(
                rstd_shape,
                tt::tt_metal::TensorLayout(
                    tensor_args.input.dtype(), tt::tt_metal::Layout::TILE, tensor_args.input.memory_config()));
        }
    }

    return output_specs;
}

tensor_return_value_t LayerNormForwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    tensor_return_value_t output_tensors;
    output_tensors.reserve(3U);

    spec_return_value_t output_specs = compute_output_specs(args, tensor_args);

    // output
    if (tensor_args.preallocated_output.has_value()) {
        output_tensors.push_back(tensor_args.preallocated_output);
    } else {
        output_tensors.push_back(create_device_tensor(output_specs[0], tensor_args.input.device()));
    }

    // mean (optional)
    if (args.return_mean_rstd) {
        if (tensor_args.preallocated_mean.has_value()) {
            output_tensors.push_back(tensor_args.preallocated_mean);
        } else {
            output_tensors.push_back(create_device_tensor(output_specs[1], tensor_args.input.device()));
        }

        // rstd (optional)
        if (tensor_args.preallocated_rstd.has_value()) {
            output_tensors.push_back(tensor_args.preallocated_rstd);
        } else {
            output_tensors.push_back(create_device_tensor(output_specs[2], tensor_args.input.device()));
        }
    } else {
        output_tensors.push_back(std::nullopt);
        output_tensors.push_back(std::nullopt);
    }

    return output_tensors;
}

ttsl::hash::hash_t LayerNormForwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& input_logical_shape = input_tensor.logical_shape();
    auto program_factory = select_program_factory(args, tensor_args);
    tt::tt_metal::operation::Hash hash = tt::tt_metal::operation::hash_operation<LayerNormForwardDeviceOperation>(
        args.epsilon, args.return_mean_rstd, program_factory.index(), input_tensor.dtype(), input_logical_shape);

    return hash;
}

}  // namespace ttml::metal::ops::layernorm_fw::device

namespace ttnn::prim {

ttml::metal::ops::layernorm_fw::device::LayerNormForwardDeviceOperation::tensor_return_value_t ttml_layernorm_fw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& beta_tensor,
    float epsilon,
    bool return_mean_rstd,
    const std::optional<ttnn::Tensor>& preallocated_output,
    const std::optional<ttnn::Tensor>& preallocated_mean,
    const std::optional<ttnn::Tensor>& preallocated_rstd) {
    using OperationType = ttml::metal::ops::layernorm_fw::device::LayerNormForwardDeviceOperation;

    auto operation_attributes =
        OperationType::operation_attributes_t{.epsilon = epsilon, .return_mean_rstd = return_mean_rstd};
    auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .gamma = gamma_tensor,
        .beta = beta_tensor,
        .preallocated_output = preallocated_output,
        .preallocated_mean = preallocated_mean,
        .preallocated_rstd = preallocated_rstd,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
