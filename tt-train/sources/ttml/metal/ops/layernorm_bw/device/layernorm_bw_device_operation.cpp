// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_bw_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "layernorm_bw_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::layernorm_bw::device {

LayerNormBackwardDeviceOperation::program_factory_t LayerNormBackwardDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return LayerNormBackwardProgramFactory{};
}

void LayerNormBackwardDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void LayerNormBackwardDeviceOperation::validate_on_program_cache_miss(
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
    const auto& mean_tensor = tensor_args.mean;
    const auto& rstd_tensor = tensor_args.rstd;
    const auto& dL_dout_tensor = tensor_args.dL_dout;
    const auto& preallocated_dx_tensor = tensor_args.preallocated_dx;
    const auto& preallocated_dgamma_components_tensor = tensor_args.preallocated_dgamma_components;
    const auto& preallocated_dbeta_components_tensor = tensor_args.preallocated_dbeta_components;

    check_tensor(input_tensor, "Input");
    check_tensor(gamma_tensor, "Gamma");
    check_tensor(mean_tensor, "Mean");
    check_tensor(rstd_tensor, "Rstd");
    check_tensor(dL_dout_tensor, "dL_dout");
    if (preallocated_dx_tensor.has_value()) {
        check_tensor(preallocated_dx_tensor.value(), "Preallocated dx");
    }
    if (preallocated_dgamma_components_tensor.has_value()) {
        check_tensor(preallocated_dgamma_components_tensor.value(), "Preallocated dgamma_components");
    }
    if (preallocated_dbeta_components_tensor.has_value()) {
        check_tensor(preallocated_dbeta_components_tensor.value(), "Preallocated dbeta_components");
    }
}

spec_return_value_t LayerNormBackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    spec_return_value_t output_specs;
    output_specs.reserve(3U);

    // dx (input gradient) - same shape as input
    if (tensor_args.preallocated_dx.has_value()) {
        output_specs.push_back(tensor_args.preallocated_dx->tensor_spec());
    } else {
        output_specs.emplace_back(
            tensor_args.input.logical_shape(),
            tt::tt_metal::TensorLayout(
                tensor_args.input.dtype(), tt::tt_metal::Layout::TILE, tensor_args.input.memory_config()));
    }

    // dgamma_components - same shape as input (will be reduced later)
    if (tensor_args.preallocated_dgamma_components.has_value()) {
        output_specs.push_back(tensor_args.preallocated_dgamma_components->tensor_spec());
    } else {
        output_specs.emplace_back(
            tensor_args.input.logical_shape(),
            tt::tt_metal::TensorLayout(
                tensor_args.gamma.dtype(), tt::tt_metal::Layout::TILE, tensor_args.gamma.memory_config()));
    }

    // dbeta_components - same shape as input (will be reduced later)
    if (tensor_args.preallocated_dbeta_components.has_value()) {
        output_specs.push_back(tensor_args.preallocated_dbeta_components->tensor_spec());
    } else {
        output_specs.emplace_back(
            tensor_args.input.logical_shape(),
            tt::tt_metal::TensorLayout(
                tensor_args.gamma.dtype(), tt::tt_metal::Layout::TILE, tensor_args.gamma.memory_config()));
    }

    return output_specs;
}

tensor_return_value_t LayerNormBackwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    tensor_return_value_t output_tensors;
    output_tensors.reserve(3U);

    spec_return_value_t output_specs = compute_output_specs(args, tensor_args);

    // dx
    if (tensor_args.preallocated_dx.has_value()) {
        output_tensors.push_back(tensor_args.preallocated_dx.value());
    } else {
        output_tensors.push_back(create_device_tensor(output_specs[0], tensor_args.input.device()));
    }

    // dgamma_components
    if (tensor_args.preallocated_dgamma_components.has_value()) {
        output_tensors.push_back(tensor_args.preallocated_dgamma_components.value());
    } else {
        output_tensors.push_back(create_device_tensor(output_specs[1], tensor_args.gamma.device()));
    }

    // dbeta_components
    if (tensor_args.preallocated_dbeta_components.has_value()) {
        output_tensors.push_back(tensor_args.preallocated_dbeta_components.value());
    } else {
        output_tensors.push_back(create_device_tensor(output_specs[2], tensor_args.gamma.device()));
    }

    return output_tensors;
}

ttsl::hash::hash_t LayerNormBackwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& input_logical_shape = input_tensor.logical_shape();
    auto program_factory = select_program_factory(args, tensor_args);
    tt::tt_metal::operation::Hash hash = tt::tt_metal::operation::hash_operation<LayerNormBackwardDeviceOperation>(
        program_factory.index(), input_tensor.dtype(), input_logical_shape);

    return hash;
}

}  // namespace ttml::metal::ops::layernorm_bw::device

namespace ttnn::prim {

ttml::metal::ops::layernorm_bw::device::LayerNormBackwardDeviceOperation::tensor_return_value_t ttml_layernorm_bw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const ttnn::Tensor& mean_tensor,
    const ttnn::Tensor& rstd_tensor,
    const ttnn::Tensor& dL_dout_tensor,
    const std::optional<ttnn::Tensor>& preallocated_dx,
    const std::optional<ttnn::Tensor>& preallocated_dgamma_components,
    const std::optional<ttnn::Tensor>& preallocated_dbeta_components) {
    using OperationType = ttml::metal::ops::layernorm_bw::device::LayerNormBackwardDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{};
    auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .gamma = gamma_tensor,
        .mean = mean_tensor,
        .rstd = rstd_tensor,
        .dL_dout = dL_dout_tensor,
        .preallocated_dx = preallocated_dx,
        .preallocated_dgamma_components = preallocated_dgamma_components,
        .preallocated_dbeta_components = preallocated_dbeta_components,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
