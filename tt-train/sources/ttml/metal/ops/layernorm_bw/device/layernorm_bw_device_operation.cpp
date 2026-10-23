// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_bw_device_operation.hpp"

#include "layernorm_bw_program_factory.hpp"
#include "metal/common/tensor_validation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::layernorm_bw::device {

void LayerNormBackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& gamma_tensor = tensor_args.gamma;
    const auto& mean_tensor = tensor_args.mean;
    const auto& rstd_tensor = tensor_args.rstd;
    const auto& dL_dout_tensor = tensor_args.dL_dout;
    const auto& preallocated_dx_tensor = tensor_args.preallocated_dx;
    const auto& preallocated_dgamma_components_tensor = tensor_args.preallocated_dgamma_components;
    const auto& preallocated_dbeta_components_tensor = tensor_args.preallocated_dbeta_components;

    check_device_tensor(input_tensor, "LayerNormBackward", "Input", {.buffer_type = tt::tt_metal::BufferType::DRAM});
    check_device_tensor(gamma_tensor, "LayerNormBackward", "Gamma", {.buffer_type = tt::tt_metal::BufferType::DRAM});
    check_device_tensor(mean_tensor, "LayerNormBackward", "Mean", {.buffer_type = tt::tt_metal::BufferType::DRAM});
    check_device_tensor(rstd_tensor, "LayerNormBackward", "Rstd", {.buffer_type = tt::tt_metal::BufferType::DRAM});
    check_device_tensor(
        dL_dout_tensor, "LayerNormBackward", "dL_dout", {.buffer_type = tt::tt_metal::BufferType::DRAM});
    TT_FATAL(
        args.compute_grad_input || args.compute_grad_gamma || args.compute_grad_beta,
        "LayerNormBackward: at least one of compute_grad_input, compute_grad_gamma and compute_grad_beta must be set");
    TT_FATAL(
        args.compute_grad_input || !preallocated_dx_tensor.has_value(),
        "LayerNormBackward: preallocated dx given but compute_grad_input is false");
    TT_FATAL(
        args.compute_grad_gamma || !preallocated_dgamma_components_tensor.has_value(),
        "LayerNormBackward: preallocated dgamma_components given but compute_grad_gamma is false");
    TT_FATAL(
        args.compute_grad_beta || !preallocated_dbeta_components_tensor.has_value(),
        "LayerNormBackward: preallocated dbeta_components given but compute_grad_beta is false");
    if (preallocated_dx_tensor.has_value()) {
        check_device_tensor(
            preallocated_dx_tensor.value(),
            "LayerNormBackward",
            "Preallocated dx",
            {.buffer_type = tt::tt_metal::BufferType::DRAM});
    }
    if (preallocated_dgamma_components_tensor.has_value()) {
        check_device_tensor(
            preallocated_dgamma_components_tensor.value(),
            "LayerNormBackward",
            "Preallocated dgamma_components",
            {.buffer_type = tt::tt_metal::BufferType::DRAM});
    }
    if (preallocated_dbeta_components_tensor.has_value()) {
        check_device_tensor(
            preallocated_dbeta_components_tensor.value(),
            "LayerNormBackward",
            "Preallocated dbeta_components",
            {.buffer_type = tt::tt_metal::BufferType::DRAM});
    }
}

spec_return_value_t LayerNormBackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // All three slots are always present so that indices stay [dx, dgamma_components, dbeta_components].
    spec_return_value_t output_specs(3U, std::nullopt);

    // dx (input gradient) - same shape as input
    if (args.compute_grad_input) {
        if (tensor_args.preallocated_dx.has_value()) {
            output_specs[0] = tensor_args.preallocated_dx->tensor_spec();
        } else {
            output_specs[0] = tt::tt_metal::TensorSpec(
                tensor_args.input.logical_shape(),
                tt::tt_metal::TensorLayout(
                    tensor_args.input.dtype(), tt::tt_metal::Layout::TILE, tensor_args.input.memory_config()));
        }
    }

    // dgamma_components - same shape as input (will be reduced later)
    if (args.compute_grad_gamma) {
        if (tensor_args.preallocated_dgamma_components.has_value()) {
            output_specs[1] = tensor_args.preallocated_dgamma_components->tensor_spec();
        } else {
            output_specs[1] = tt::tt_metal::TensorSpec(
                tensor_args.input.logical_shape(),
                tt::tt_metal::TensorLayout(
                    tensor_args.gamma.dtype(), tt::tt_metal::Layout::TILE, tensor_args.gamma.memory_config()));
        }
    }

    // dbeta_components - same shape as input (will be reduced later)
    if (args.compute_grad_beta) {
        if (tensor_args.preallocated_dbeta_components.has_value()) {
            output_specs[2] = tensor_args.preallocated_dbeta_components->tensor_spec();
        } else {
            output_specs[2] = tt::tt_metal::TensorSpec(
                tensor_args.input.logical_shape(),
                tt::tt_metal::TensorLayout(
                    tensor_args.gamma.dtype(), tt::tt_metal::Layout::TILE, tensor_args.gamma.memory_config()));
        }
    }

    return output_specs;
}

tensor_return_value_t LayerNormBackwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    tensor_return_value_t output_tensors;
    output_tensors.reserve(3U);

    spec_return_value_t output_specs = compute_output_specs(args, tensor_args);

    const auto create_output = [&](const std::optional<tt::tt_metal::TensorSpec>& spec,
                                   const std::optional<ttnn::Tensor>& preallocated,
                                   const ttnn::Tensor& device_source) -> std::optional<ttnn::Tensor> {
        if (!spec.has_value()) {
            return std::nullopt;
        }
        if (preallocated.has_value()) {
            return preallocated.value();
        }
        return ttnn::create_device_tensor(*spec, device_source.device());
    };

    // dx
    output_tensors.push_back(create_output(output_specs[0], tensor_args.preallocated_dx, tensor_args.input));
    // dgamma_components
    output_tensors.push_back(
        create_output(output_specs[1], tensor_args.preallocated_dgamma_components, tensor_args.gamma));
    // dbeta_components
    output_tensors.push_back(
        create_output(output_specs[2], tensor_args.preallocated_dbeta_components, tensor_args.gamma));

    return output_tensors;
}

ttsl::hash::hash_t LayerNormBackwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& input_logical_shape = input_tensor.logical_shape();
    tt::tt_metal::operation::Hash hash = tt::tt_metal::operation::hash_operation<LayerNormBackwardDeviceOperation>(
        args, input_tensor.dtype(), input_logical_shape);

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
    const std::optional<ttnn::Tensor>& preallocated_dbeta_components,
    bool compute_grad_input,
    bool compute_grad_gamma,
    bool compute_grad_beta) {
    using OperationType = ttml::metal::ops::layernorm_bw::device::LayerNormBackwardDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .compute_grad_input = compute_grad_input,
        .compute_grad_gamma = compute_grad_gamma,
        .compute_grad_beta = compute_grad_beta};
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
