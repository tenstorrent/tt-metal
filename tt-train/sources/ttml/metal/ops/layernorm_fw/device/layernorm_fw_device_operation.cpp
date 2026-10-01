// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_fw_device_operation.hpp"

#include <enchantum/enchantum.hpp>
#include <tt-metalium/constants.hpp>

#include "layernorm_fw_program_factory.hpp"
#include "metal/ops/layernorm_common.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::layernorm_fw::device {

void LayerNormForwardDeviceOperation::validate_on_program_cache_miss(
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
    const auto& beta_tensor = tensor_args.beta;
    const auto& preallocated_output_tensor = tensor_args.preallocated_output;
    const auto& preallocated_mean_tensor = tensor_args.preallocated_mean;
    const auto& preallocated_rstd_tensor = tensor_args.preallocated_rstd;

    check_tensor(input_tensor, "Input");
    check_tensor(gamma_tensor, "Gamma");
    check_tensor(beta_tensor, "Beta");

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
        beta_tensor.logical_shape() == parameter_shape,
        "Beta tensor must have shape {}. Got shape {}",
        parameter_shape,
        beta_tensor.logical_shape());
    TT_FATAL(
        gamma_tensor.padded_shape()[-1] == input_tensor.padded_shape()[-1],
        "Gamma padded width must match the input padded width. Got {} and {}",
        gamma_tensor.padded_shape()[-1],
        input_tensor.padded_shape()[-1]);
    TT_FATAL(
        beta_tensor.padded_shape()[-1] == input_tensor.padded_shape()[-1],
        "Beta padded width must match the input padded width. Got {} and {}",
        beta_tensor.padded_shape()[-1],
        input_tensor.padded_shape()[-1]);

    const auto expected_output_spec = tt::tt_metal::TensorSpec(input_shape, input_tensor.tensor_spec().tensor_layout());
    if (preallocated_output_tensor.has_value()) {
        check_tensor(preallocated_output_tensor.value(), "Preallocated output");
        TT_FATAL(
            preallocated_output_tensor->tensor_spec() == expected_output_spec,
            "Preallocated output TensorSpec must match the input TensorSpec");
    }
    if (args.return_mean_rstd) {
        if (preallocated_mean_tensor.has_value()) {
            check_tensor(preallocated_mean_tensor.value(), "Preallocated mean");
            layernorm_common::validate_stats_geometry(*preallocated_mean_tensor, input_tensor, "Preallocated mean");
        }
        if (preallocated_rstd_tensor.has_value()) {
            check_tensor(preallocated_rstd_tensor.value(), "Preallocated rstd");
            layernorm_common::validate_stats_geometry(*preallocated_rstd_tensor, input_tensor, "Preallocated rstd");
        }
    } else {
        TT_FATAL(
            !preallocated_mean_tensor.has_value() && !preallocated_rstd_tensor.has_value(),
            "Preallocated mean/rstd tensors require return_mean_rstd=true");
    }
}

spec_return_value_t LayerNormForwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    spec_return_value_t output_specs;
    output_specs.reserve(3U);

    // output - same shape as input
    auto input_shape = tensor_args.input.logical_shape();
    const auto& input_layout = tensor_args.input.tensor_spec().tensor_layout();
    output_specs.emplace_back(input_shape, input_layout);

    // mean - shape is [B, 1, S, 1]
    if (args.return_mean_rstd) {
        output_specs.push_back(layernorm_common::stats_tensor_spec(tensor_args.input));

        // rstd - same shape as mean [B, H, S, 1]
        output_specs.push_back(layernorm_common::stats_tensor_spec(tensor_args.input));
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
        output_tensors.push_back(ttnn::create_device_tensor(output_specs[0], tensor_args.input.device()));
    }

    // mean (optional)
    if (args.return_mean_rstd) {
        if (tensor_args.preallocated_mean.has_value()) {
            output_tensors.push_back(tensor_args.preallocated_mean);
        } else {
            output_tensors.push_back(ttnn::create_device_tensor(output_specs[1], tensor_args.input.device()));
        }

        // rstd (optional)
        if (tensor_args.preallocated_rstd.has_value()) {
            output_tensors.push_back(tensor_args.preallocated_rstd);
        } else {
            output_tensors.push_back(ttnn::create_device_tensor(output_specs[2], tensor_args.input.device()));
        }
    } else {
        output_tensors.push_back(std::nullopt);
        output_tensors.push_back(std::nullopt);
    }

    return output_tensors;
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
