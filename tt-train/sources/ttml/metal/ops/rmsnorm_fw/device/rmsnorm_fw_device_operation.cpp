// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_fw_device_operation.hpp"

#include <enchantum/enchantum.hpp>
#include <tt-metalium/constants.hpp>

#include "rmsnorm_fw_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::rmsnorm_fw::device {

void RMSNormForwardDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto* expected_device = tensor_args.input.device();
    auto check_tensor = [expected_device](const ttnn::Tensor& tensor, const std::string& name) {
        TT_FATAL(
            tensor.storage_type() == ttnn::StorageType::DEVICE,
            "RMSNormForward operation requires {} to be on Device. Input storage type: {}",
            name,
            enchantum::to_string(tensor.storage_type()));

        TT_FATAL(
            tensor.buffer() != nullptr,
            "Operands to RMSNormForward need to be allocated in buffers on the device. Buffer is null. Tensor name {}",
            name);

        TT_FATAL(
            tensor.layout() == tt::tt_metal::Layout::TILE,
            "RMSNormForward operation requires tensor to be in Tile layout. {} tensor layout: {}",
            name,
            enchantum::to_string(tensor.layout()));

        TT_FATAL(
            tensor.dtype() == tt::tt_metal::DataType::BFLOAT16,
            "RMSNormForward operation requires tensor to be of BFLOAT16 data type. {} tensor data type: {}",
            name,
            enchantum::to_string(tensor.dtype()));

        TT_FATAL(
            tensor.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
            "RMSNormForward operation requires Interleaved memory layout. {} "
            "memory layout: `{}`",
            name,
            enchantum::to_string(tensor.memory_config().memory_layout()));

        TT_FATAL(
            tensor.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM,
            "RMSNormForward operation requires {} to be in DRAM. Buffer type: {}",
            name,
            enchantum::to_string(tensor.memory_config().buffer_type()));

        const auto tile = tensor.tensor_spec().tile();
        TT_FATAL(
            tile == tt::tt_metal::Tile{} && !tile.get_transpose_within_face() && !tile.get_transpose_of_faces(),
            "RMSNormForward operation requires {} to use the canonical non-transposed 32x32 tile",
            name);

        TT_FATAL(
            tensor.device() == expected_device,
            "RMSNormForward operation requires {} to be on the same MeshDevice as Input",
            name);
    };

    const auto& input_tensor = tensor_args.input;
    const auto& gamma_tensor = tensor_args.gamma;
    const auto& preallocated_rms_tensor = tensor_args.preallocated_rms;
    const auto& preallocated_output_tensor = tensor_args.preallocated_output;
    check_tensor(input_tensor, "Input");
    check_tensor(gamma_tensor, "Gamma");

    const auto& input_shape = input_tensor.logical_shape();
    const auto& input_padded_shape = input_tensor.padded_shape();
    TT_FATAL(input_shape.rank() == 4U, "RMSNormForward input must be 4D [B, N, S, C], got {}", input_shape);
    TT_FATAL(
        input_shape[0] > 0U && input_shape[1] > 0U && input_shape[2] > 0U && input_shape[3] > 0U,
        "RMSNormForward input dimensions must be nonzero, got {}",
        input_shape);
    TT_FATAL(
        input_padded_shape.rank() == 4U && input_padded_shape[0] == input_shape[0] &&
            input_padded_shape[1] == input_shape[1],
        "RMSNormForward input may be overpadded in height but not in leading dimensions. Logical shape: {}, padded "
        "shape: {}",
        input_shape,
        input_padded_shape);

    const uint32_t expected_padded_width =
        ((input_shape[-1] + tt::constants::TILE_WIDTH - 1U) / tt::constants::TILE_WIDTH) * tt::constants::TILE_WIDTH;
    TT_FATAL(
        input_padded_shape[-1] == expected_padded_width,
        "RMSNormForward input must use canonical width padding {}. Got {}",
        expected_padded_width,
        input_padded_shape[-1]);
    const auto& input_alignment = input_tensor.tensor_spec().tensor_layout().get_alignment();
    TT_FATAL(
        !input_alignment.empty() && input_alignment[-1] == tt::constants::TILE_WIDTH,
        "RMSNormForward input width alignment must be {}. Got {}",
        tt::constants::TILE_WIDTH,
        input_alignment);

    const auto expected_gamma_shape = ttnn::Shape({1U, 1U, 1U, input_shape[-1]});
    TT_FATAL(
        gamma_tensor.logical_shape() == expected_gamma_shape,
        "RMSNormForward Gamma must have shape {}. Got {}",
        expected_gamma_shape,
        gamma_tensor.logical_shape());
    TT_FATAL(
        gamma_tensor.padded_shape()[-1] == input_padded_shape[-1],
        "RMSNormForward Gamma padded width must match Input. Got {} and {}",
        gamma_tensor.padded_shape()[-1],
        input_padded_shape[-1]);

    const auto& input_spec = input_tensor.tensor_spec();
    auto rms_shape = input_shape;
    rms_shape[-1] = 1U;
    const auto expected_rms_spec = tt::tt_metal::TensorSpec(rms_shape, input_spec.tensor_layout());
    if (preallocated_rms_tensor.has_value()) {
        check_tensor(preallocated_rms_tensor.value(), "Preallocated RMS");
        TT_FATAL(args.return_intermediates, "RMSNormForward preallocated RMS requires return_intermediates=true");
        TT_FATAL(
            preallocated_rms_tensor->tensor_spec() == expected_rms_spec,
            "RMSNormForward preallocated RMS TensorSpec must match the input-derived RMS TensorSpec");
    }
    if (preallocated_output_tensor.has_value()) {
        check_tensor(preallocated_output_tensor.value(), "Preallocated Output");
        TT_FATAL(
            preallocated_output_tensor->tensor_spec() == input_spec,
            "RMSNormForward preallocated Output TensorSpec must match Input TensorSpec");
    }
}

spec_return_value_t RMSNormForwardDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    spec_return_value_t output_specs;
    output_specs.reserve(1U + static_cast<uint32_t>(args.return_intermediates));

    const auto& input_spec = tensor_args.input.tensor_spec();
    const auto& input_layout = input_spec.tensor_layout();
    output_specs.push_back(input_spec);

    if (args.return_intermediates) {
        auto shape = tensor_args.input.logical_shape();
        shape[-1] = 1U;  // RMS is a scalar per row
        output_specs.emplace_back(shape, input_layout);
    }

    return output_specs;
}

tensor_return_value_t RMSNormForwardDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    tensor_return_value_t output_tensors;
    output_tensors.reserve(1U + static_cast<uint32_t>(args.return_intermediates));

    spec_return_value_t output_specs = compute_output_specs(args, tensor_args);

    if (tensor_args.preallocated_output.has_value()) {
        output_tensors.push_back(tensor_args.preallocated_output.value());
    } else {
        output_tensors.push_back(ttnn::create_device_tensor(output_specs[0], tensor_args.input.device()));
    }

    if (args.return_intermediates) {
        if (tensor_args.preallocated_rms.has_value()) {
            output_tensors.push_back(tensor_args.preallocated_rms.value());
        } else {
            output_tensors.push_back(ttnn::create_device_tensor(output_specs[1], tensor_args.input.device()));
        }
    }

    return output_tensors;
}

}  // namespace ttml::metal::ops::rmsnorm_fw::device

namespace ttnn::prim {

ttml::metal::ops::rmsnorm_fw::device::RMSNormForwardDeviceOperation::tensor_return_value_t ttml_rmsnorm_fw(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& gamma_tensor,
    bool return_intermediates,
    float epsilon,
    const std::optional<ttnn::Tensor>& preallocated_rms,
    const std::optional<ttnn::Tensor>& preallocated_output) {
    using OperationType = ttml::metal::ops::rmsnorm_fw::device::RMSNormForwardDeviceOperation;

    auto operation_attributes =
        OperationType::operation_attributes_t{.return_intermediates = return_intermediates, .epsilon = epsilon};
    auto tensor_args = OperationType::tensor_args_t{
        .input = input_tensor,
        .gamma = gamma_tensor,
        .preallocated_rms = preallocated_rms,
        .preallocated_output = preallocated_output,
    };

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
