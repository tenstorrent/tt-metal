// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_bw_device_operation.hpp"

#include <enchantum/enchantum.hpp>

#include "rmsnorm_bw_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::rmsnorm_bw::device {

namespace {

void check_device_tensor(const ttnn::Tensor& tensor, const std::string& name, tt::tt_metal::DataType dtype) {
    TT_FATAL(
        tensor.storage_type() == ttnn::StorageType::DEVICE,
        "RMSNormBackward operation requires {} to be on Device. Input storage type: {}",
        name,
        enchantum::to_string(tensor.storage_type()));

    TT_FATAL(
        tensor.buffer() != nullptr,
        "Operands to RMSNormBackward need to be allocated in buffers on the device. Buffer is null. Tensor name {}",
        name);

    TT_FATAL(
        tensor.layout() == tt::tt_metal::Layout::TILE,
        "RMSNormBackward operation requires tensor to be in Tile layout. {} tensor layout: {}",
        name,
        enchantum::to_string(tensor.layout()));

    const auto& tile = tensor.tensor_spec().tile();
    TT_FATAL(
        tile.get_height() == tt::constants::TILE_HEIGHT && tile.get_width() == tt::constants::TILE_WIDTH,
        "RMSNormBackward operation requires 32x32 tiles. {} tile: {}x{}",
        name,
        tile.get_height(),
        tile.get_width());

    TT_FATAL(
        tensor.dtype() == dtype,
        "RMSNormBackward operation requires {} to be of {} data type, got {}",
        name,
        enchantum::to_string(dtype),
        enchantum::to_string(tensor.dtype()));

    TT_FATAL(
        tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "RMSNormBackward operation requires Interleaved memory layout. {} "
        "memory layout: `{}`",
        name,
        enchantum::to_string(tensor.memory_config().memory_layout()));
}

// rms and the external sums hold one value per row of the input, so their row grid must match the input's.
void check_per_row_tensor(const ttnn::Tensor& tensor, const ttnn::Tensor& input, const std::string& name) {
    const auto& padded = tensor.padded_shape();
    const auto& input_padded = input.padded_shape();
    TT_FATAL(
        padded.rank() == 4U && padded[0] == input_padded[0] && padded[1] == input_padded[1] &&
            padded[2] == input_padded[2],
        "{} padded shape {} must match the first three padded dims of the input {}",
        name,
        padded,
        input_padded);
}

}  // namespace

void RMSNormBackwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    constexpr auto bf16 = tt::tt_metal::DataType::BFLOAT16;
    const auto& input_tensor = tensor_args.input;

    check_device_tensor(input_tensor, "Input", bf16);
    check_device_tensor(tensor_args.gamma, "Gamma", bf16);
    check_device_tensor(tensor_args.dL_dout, "dL_dout", bf16);
    TT_FATAL(
        tensor_args.dL_dout.padded_shape() == input_tensor.padded_shape(),
        "dL_dout padded shape {} must match the input padded shape {}",
        tensor_args.dL_dout.padded_shape(),
        input_tensor.padded_shape());

    if (args.mode == RMSNormBackwardMode::LocalSum) {
        TT_FATAL(
            !tensor_args.preallocated_da.has_value() && !tensor_args.preallocated_dgamma_components.has_value(),
            "RMSNormBackward LocalSum mode does not take preallocated gradients");
    } else {
        TT_FATAL(tensor_args.rms.has_value(), "RMSNormBackward requires rms outside LocalSum mode");
        check_device_tensor(tensor_args.rms.value(), "RMS", bf16);
        check_per_row_tensor(tensor_args.rms.value(), input_tensor, "RMS");
        if (tensor_args.preallocated_da.has_value()) {
            check_device_tensor(tensor_args.preallocated_da.value(), "Preallocated dL_da", bf16);
        }
        if (tensor_args.preallocated_dgamma_components.has_value()) {
            TT_FATAL(
                !compute_layout(args, tensor_args).dgamma_on_chip,
                "RMSNormBackward accumulates dgamma on chip for this shape and cannot write into preallocated "
                "dL_dgamma_components");
            check_device_tensor(
                tensor_args.preallocated_dgamma_components.value(), "Preallocated dL_dgamma_components", bf16);
        }
    }

    if (args.mode == RMSNormBackwardMode::ExternalScale) {
        TT_FATAL(tensor_args.external_sums.has_value(), "RMSNormBackward ExternalScale mode requires external_sums");
        const auto& sums = tensor_args.external_sums.value();
        check_device_tensor(sums, "External sums", tt::tt_metal::DataType::FLOAT32);
        check_per_row_tensor(sums, input_tensor, "External sums");
        TT_FATAL(
            sums.logical_shape()[-1] % tt::constants::TILE_WIDTH == 0U,
            "External sums last dim {} must hold one tile per shard",
            sums.logical_shape()[-1]);
        TT_FATAL(
            args.full_width >= input_tensor.logical_shape()[-1],
            "full_width {} must be at least the local width {}",
            args.full_width,
            input_tensor.logical_shape()[-1]);
    } else {
        TT_FATAL(!tensor_args.external_sums.has_value(), "external_sums is only used by ExternalScale mode");
    }
}

spec_return_value_t RMSNormBackwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    spec_return_value_t output_specs;

    if (args.mode == RMSNormBackwardMode::LocalSum) {
        auto shape = input.logical_shape();
        shape[-1] = tt::constants::TILE_WIDTH;
        output_specs.emplace_back(
            shape,
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::FLOAT32, tt::tt_metal::Layout::TILE, input.memory_config()));
        return output_specs;
    }

    output_specs.reserve(2U);
    if (tensor_args.preallocated_da.has_value()) {
        output_specs.push_back(tensor_args.preallocated_da->tensor_spec());
    } else {
        output_specs.emplace_back(
            input.logical_shape(),
            tt::tt_metal::TensorLayout(input.dtype(), tt::tt_metal::Layout::TILE, input.memory_config()));
    }

    const auto layout = compute_layout(args, tensor_args);
    if (layout.dgamma_on_chip) {
        output_specs.emplace_back(
            ttnn::Shape({1U, 1U, layout.num_cores * tt::constants::TILE_HEIGHT, input.logical_shape()[-1]}),
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::FLOAT32, tt::tt_metal::Layout::TILE, tensor_args.gamma.memory_config()));
    } else if (tensor_args.preallocated_dgamma_components.has_value()) {
        output_specs.push_back(tensor_args.preallocated_dgamma_components->tensor_spec());
    } else {
        output_specs.emplace_back(
            input.logical_shape(),
            tt::tt_metal::TensorLayout(
                tensor_args.gamma.dtype(), tt::tt_metal::Layout::TILE, tensor_args.gamma.memory_config()));
    }
    return output_specs;
}

tensor_return_value_t RMSNormBackwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const spec_return_value_t output_specs = compute_output_specs(args, tensor_args);
    auto* device = tensor_args.input.device();

    tensor_return_value_t output_tensors;
    output_tensors.reserve(output_specs.size());
    if (args.mode == RMSNormBackwardMode::LocalSum) {
        output_tensors.push_back(ttnn::create_device_tensor(output_specs[0], device));
        return output_tensors;
    }

    if (tensor_args.preallocated_da.has_value()) {
        output_tensors.push_back(tensor_args.preallocated_da.value());
    } else {
        output_tensors.push_back(ttnn::create_device_tensor(output_specs[0], device));
    }

    const bool dgamma_on_chip = compute_layout(args, tensor_args).dgamma_on_chip;
    if (!dgamma_on_chip && tensor_args.preallocated_dgamma_components.has_value()) {
        output_tensors.push_back(tensor_args.preallocated_dgamma_components.value());
    } else {
        output_tensors.push_back(ttnn::create_device_tensor(output_specs[1], device));
    }
    return output_tensors;
}

ttsl::hash::hash_t RMSNormBackwardDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const uint32_t num_external_tiles = tensor_args.external_sums.has_value()
                                            ? tensor_args.external_sums->padded_shape()[-1] / tt::constants::TILE_WIDTH
                                            : 0U;
    return tt::tt_metal::operation::hash_operation<RMSNormBackwardDeviceOperation>(
        args.mode, args.full_width, input_tensor.dtype(), input_tensor.logical_shape(), num_external_tiles);
}

}  // namespace ttml::metal::ops::rmsnorm_bw::device

namespace ttnn::prim {

ttml::metal::ops::rmsnorm_bw::device::RMSNormBackwardDeviceOperation::tensor_return_value_t ttml_rmsnorm_bw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    const std::optional<ttnn::Tensor>& rms_tensor,
    const ttnn::Tensor& dL_dout_tensor,
    float epsilon,
    const std::optional<ttnn::Tensor>& preallocated_da,
    const std::optional<ttnn::Tensor>& preallocated_dgamma_components,
    ttml::metal::ops::rmsnorm_bw::device::RMSNormBackwardMode mode,
    uint32_t full_width,
    const std::optional<ttnn::Tensor>& external_sums) {
    using OperationType = ttml::metal::ops::rmsnorm_bw::device::RMSNormBackwardDeviceOperation;

    auto operation_attributes =
        OperationType::operation_attributes_t{.epsilon = epsilon, .mode = mode, .full_width = full_width};
    auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .gamma = gamma_tensor,
        .rms = rms_tensor,
        .dL_dout = dL_dout_tensor,
        .external_sums = external_sums,
        .preallocated_da = preallocated_da,
        .preallocated_dgamma_components = preallocated_dgamma_components,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
