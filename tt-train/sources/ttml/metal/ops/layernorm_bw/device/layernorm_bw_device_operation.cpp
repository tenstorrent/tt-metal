// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_bw_device_operation.hpp"

#include <enchantum/enchantum.hpp>
#include <tt-metalium/constants.hpp>

#include "layernorm_bw_program_factory.hpp"
#include "metal/ops/layernorm_common.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::layernorm_bw::device {

void LayerNormBackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto* expected_device = tensor_args.input.device();
    auto check_tensor = [expected_device](const ttnn::Tensor& tensor, const std::string& name) {
        TT_FATAL(
            tensor.storage_type() == ttnn::StorageType::DEVICE,
            "Tensor's '{}' storage type must be {}. Got storage type: {}",
            name,
            enchantum::to_string(ttnn::StorageType::DEVICE),
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
            tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
            "Tensor '{}' must use Interleaved memory layout. Got memory layout: {}",
            name,
            enchantum::to_string(tensor.memory_config().memory_layout()));

        TT_FATAL(
            tensor.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM,
            "Tensor '{}' memory config must use DRAM. Got buffer type: {}",
            name,
            enchantum::to_string(tensor.memory_config().buffer_type()));

        const auto tile = tensor.tensor_spec().tile();
        TT_FATAL(
            tile == tt::tt_metal::Tile{} && !tile.get_transpose_within_face() && !tile.get_transpose_of_faces(),
            "Tensor '{}' must use the canonical non-transposed 32x32 tile",
            name);

        TT_FATAL(
            tensor.device() == expected_device,
            "Tensor '{}' must be allocated on the same MeshDevice as the input tensor",
            name);
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

    const auto& input_shape = input_tensor.logical_shape();
    TT_FATAL(input_shape.rank() == 4U, "Input tensor must be 4D [B, N, S, C], got shape {}", input_shape);
    const uint32_t expected_padded_width =
        ((input_shape[-1] + tt::constants::TILE_WIDTH - 1U) / tt::constants::TILE_WIDTH) * tt::constants::TILE_WIDTH;
    TT_FATAL(
        input_tensor.padded_shape()[-1] == expected_padded_width,
        "Input tensor may be overpadded in height but must use canonical width padding {}. Got {}",
        expected_padded_width,
        input_tensor.padded_shape()[-1]);
    const auto parameter_shape = ttnn::Shape({1U, 1U, 1U, input_shape[-1]});
    TT_FATAL(
        gamma_tensor.logical_shape() == parameter_shape,
        "Gamma tensor must have shape {}. Got shape {}",
        parameter_shape,
        gamma_tensor.logical_shape());
    TT_FATAL(
        gamma_tensor.padded_shape()[-1] == input_tensor.padded_shape()[-1],
        "Gamma padded width must match the input padded width. Got {} and {}",
        gamma_tensor.padded_shape()[-1],
        input_tensor.padded_shape()[-1]);

    layernorm_common::validate_stats_geometry(mean_tensor, input_tensor, "Mean");
    layernorm_common::validate_stats_geometry(rstd_tensor, input_tensor, "Rstd");
    TT_FATAL(
        dL_dout_tensor.tensor_spec() == input_tensor.tensor_spec(),
        "dL_dout TensorSpec must match the input TensorSpec");

    const auto& expected_output_spec = input_tensor.tensor_spec();
    if (preallocated_dx_tensor.has_value()) {
        check_tensor(preallocated_dx_tensor.value(), "Preallocated dx");
        TT_FATAL(
            preallocated_dx_tensor->tensor_spec() == expected_output_spec,
            "Preallocated dx TensorSpec must match the input TensorSpec");
    }
    if (preallocated_dgamma_components_tensor.has_value()) {
        check_tensor(preallocated_dgamma_components_tensor.value(), "Preallocated dgamma_components");
        TT_FATAL(
            preallocated_dgamma_components_tensor->tensor_spec() == expected_output_spec,
            "Preallocated dgamma_components TensorSpec must match the input TensorSpec");
    }
    if (preallocated_dbeta_components_tensor.has_value()) {
        check_tensor(preallocated_dbeta_components_tensor.value(), "Preallocated dbeta_components");
        TT_FATAL(
            preallocated_dbeta_components_tensor->tensor_spec() == expected_output_spec,
            "Preallocated dbeta_components TensorSpec must match the input TensorSpec");
    }
}

spec_return_value_t LayerNormBackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    spec_return_value_t output_specs;
    output_specs.reserve(3U);

    // dx (input gradient) - same shape as input
    const auto& input_spec = tensor_args.input.tensor_spec();
    output_specs.push_back(input_spec);

    // dgamma_components - same shape as input (will be reduced later)
    output_specs.push_back(input_spec);

    // dbeta_components - same shape as input (will be reduced later)
    output_specs.push_back(input_spec);

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
        output_tensors.push_back(ttnn::create_device_tensor(output_specs[0], tensor_args.input.device()));
    }

    // dgamma_components
    if (tensor_args.preallocated_dgamma_components.has_value()) {
        output_tensors.push_back(tensor_args.preallocated_dgamma_components.value());
    } else {
        output_tensors.push_back(ttnn::create_device_tensor(output_specs[1], tensor_args.gamma.device()));
    }

    // dbeta_components
    if (tensor_args.preallocated_dbeta_components.has_value()) {
        output_tensors.push_back(tensor_args.preallocated_dbeta_components.value());
    } else {
        output_tensors.push_back(ttnn::create_device_tensor(output_specs[2], tensor_args.gamma.device()));
    }

    return output_tensors;
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
